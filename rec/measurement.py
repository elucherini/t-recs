from abc import ABCMeta, abstractmethod
from .debug import VerboseMode
import numpy as np
import networkx as nx

class Measurement(VerboseMode, metaclass=ABCMeta):
    def __init__(self, default_increment, verbose=False):
        VerboseMode.__init__(self, __name__.upper(), verbose)
        self.measurement_data = np.zeros(default_increment)
        self._index = 0
        self._default_increment = default_increment

    def _expand_array(self, timesteps=None):    
        '''
            Internal function to expand the measurement module to include
            additional timesteps. This is called when the measurement array
            we want to modify only has a few free positions left.

            Args:
                array (:obj:`numpy.array`) : measurement array to expand
                timesteps (int, optional): number of steps (positions in the array) to add.
                    If None, the array is incremented by _default_increment.

            Returns: :obj:`numpy.array` resized array.
        '''
        if timesteps is None:
            timesteps = self._default_increment
        self.log("Expanding measurement array size to: %d" % (self.measurement_data.size + timesteps))
        return np.resize(self.measurement_data, self.measurement_data.size + timesteps)


    def _generate_interaction_histogram(self, interactions, num_users, num_items):
        ''' Internal function that returns a histogram of the number 
            of interactions per item at the given timestep.
        
            Args:
                interactions (:obj:`numpy.array`): array of user interactions.
                num_users (int): number of users in the system
                num_items (int): number of items in the system

            Returns: :obj:`numpy.array` histogram of the number of interactions
                aggregated by items at the given timestep.
        '''
        histogram = np.zeros(num_items)
        np.add.at(histogram, interactions, 1)
        # Check that there's one interaction per user
        assert(histogram.sum() == num_users)
        return histogram

    def get_measurement(self):
        if self._index > 0 and self.name is not None:
            return {self.name: self.measurement_data[:self._index]}
        else:
            return None

    @abstractmethod
    def measure(self, step, interactions, recommender):
        self._index += 1

    def get_timesteps(self):
        return self._index


class HomogeneityMeasurement(Measurement):
    def __init__(self, default_increment=100, verbose=False):
        self.histogram = None
        self._old_histogram = None
        self.name = 'homogeneity'
        Measurement.__init__(self, default_increment, verbose)

    def measure(self, step, interactions, recommender):
        '''
            Measures the homogeneity of user interactions
            (i.e., whether interactions are spread among many items or only a
            few items).

            Args:
                step (int): current time step
                interactions (:obj:`numpy.array`): non-aggregated array of interactions (i.e.,
                     array of length |U| s.t. element u is the index of the
                     item user u interacted with)
                num_users (int): number of users in the system
                num_items (int): number of items in the system
    
            Returns: :obj:`numpy.array` with homogeneity.
        '''
        if self.measurement_data is None or self._index >= self.measurement_data.size:
            self.measurement_data = self._expand_array()
        assert(interactions.size == recommender.num_users)
        histogram = self._generate_interaction_histogram(recommender.interactions, recommender.num_users,
            recommender.num_items)
        histogram[::-1].sort()
        if self._old_histogram is None:
            self._old_histogram = np.zeros(recommender.num_items)
        # delta(t) = Area(histogram(t-1)) - Area(histogram(t))
        self.measurement_data[self._index] = np.trapz(self._old_histogram, dx=1) - \
                                    np.trapz(histogram, dx=1)
        self._old_histogram = np.copy(histogram)
        self.histogram = histogram
        Measurement.measure(self, step, interactions, recommender)


class MSEMeasurement(Measurement):
    def __init__(self, default_increment=100, verbose=False):
        self.name = 'MSE'
        Measurement.__init__(self, default_increment, verbose)

    def measure(self, step, interactions, recommender):
        """ Measures and records the mean
            squared error between the user preferences predicted
            by the system and the users' actual preferences.

            Args:
                predicted (:obj:`numpy.array`): user preferences
                    predicted by the system
                actual (:obj:`numpy.array`): actual user preferences,
                    unknown to the system.
        """
        if self.measurement_data is None or self._index >= self.measurement_data.size:
            self.measurement_data = self._expand_array()

        self.measurement_data[self._index] = ((recommender.predicted_scores - recommender.actual_user_scores.actual_scores)**2).mean()
        Measurement.measure(self, step, interactions, recommender)


class DiffusionTreeMeasurement(Measurement):
    def __init__(self, infection_state, default_increment=100, verbose=False):
        self.name = 'Diffusion tree'
        self._old_infection_state = None
        self.diffusion_tree = nx.Graph()
        self._manage_new_infections(None, np.copy(infection_state))
        self._old_infection_state = np.copy(infection_state)
        Measurement.__init__(self, default_increment, verbose)

    def _find_parents(self, user_profiles, 
                                    new_infected_users):
        if (self._old_infection_state == 0).all():
            # Node is root
            return None
        # TODO: function is_following() based on code below:
        # candidates must have been previously infected
        prev_infected_users = np.where(self._old_infection_state > 0)[0]
        # candidates must be connected to newly infected users
        candidate_parents = user_profiles[:,prev_infected_users][new_infected_users]
        parents = prev_infected_users[np.argsort(candidate_parents)[:,-1]]
        return parents

    def _add_to_graph(self, user_profiles, new_infected_users):
        print("Infected users to add:\n", new_infected_users)
        self.diffusion_tree.add_nodes_from(new_infected_users)
        parents = self._find_parents(user_profiles, new_infected_users)
        # connect parent(s) and child(ren)
        if parents is not None:
            edges = np.vstack((parents, new_infected_users)).T
            self.diffusion_tree.add_edges_from(edges)

    def _manage_new_infections(self, user_profiles, current_infection_state):
        if self._old_infection_state is None:
            self._old_infection_state = np.zeros(current_infection_state.shape)
        new_infections = (current_infection_state - 
                            self._old_infection_state)
        if (new_infections == 0).all():
            # no new infections
            return
        new_infected_users = np.where(new_infections > 0)[0] # only the rows
        self._add_to_graph(user_profiles, new_infected_users)

    def measure(self, step, interactions, recommender):
        # test
        # number of infected users
        num_infected = np.sum(recommender.infection_state)
        self._manage_new_infections(recommender.user_profiles,
            recommender.infection_state)
        #print(recommender.infection_state)
        self._old_infection_state = np.copy(recommender.infection_state)