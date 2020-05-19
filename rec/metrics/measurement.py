from abc import ABC, abstractmethod
from rec.utils import VerboseMode
from rec.components import BaseObservable
import numpy as np

class Measurement(BaseObservable, VerboseMode, ABC):
    def __init__(self, verbose=False, init_value=None):
        VerboseMode.__init__(self, __name__.upper(), verbose)
        self.measurement_data = list()
        self.measurement_data.append(init_value)

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
        return self.get_observable(data=self.measurement_data)

    @abstractmethod
    def measure(self, step, interactions, recommender):
        pass

    def get_timesteps(self):
        return len(self.measurement_data)


class HomogeneityMeasurement(Measurement):
    def __init__(self, verbose=False):
        self.histogram = None
        self._old_histogram = None
        self.name = 'homogeneity'
        Measurement.__init__(self, verbose, init_value=None)

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
        assert(interactions.size == recommender.num_users)
        histogram = self._generate_interaction_histogram(interactions,
                                                         recommender.num_users,
                                                         recommender.num_items)
        histogram[::-1].sort()
        if self._old_histogram is None:
            self._old_histogram = np.zeros(recommender.num_items)
        self.measurement_data.append(np.trapz(self._old_histogram, dx=1) - \
                                    np.trapz(histogram, dx=1))
        self._old_histogram = np.copy(histogram)
        self.histogram = histogram


class MSEMeasurement(Measurement):
    def __init__(self, verbose=False):
        self.name = 'MSE'
        Measurement.__init__(self, verbose, init_value=None)

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
        self.measurement_data.append(((recommender.predicted_scores -
            recommender.actual_users.actual_user_scores)**2).mean())


class DiffusionTreeMeasurement(Measurement):
    def __init__(self, infection_state, verbose=False):
        import networkx as nx
        self.name = '# Infected'
        self._old_infection_state = None
        self.diffusion_tree = nx.Graph()
        self._manage_new_infections(None, np.copy(infection_state))
        self._old_infection_state = np.copy(infection_state)
        Measurement.__init__(self, verbose,
            init_value=self.diffusion_tree.number_of_nodes())

    def _find_parents(self, user_profiles, new_infected_users):
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
            return 0
        new_infected_users = np.where(new_infections > 0)[0] # only the rows
        self._add_to_graph(user_profiles, new_infected_users)
        # return number of new infections
        return new_infected_users.shape[0]

    def measure(self, step, interactions, recommender):
        num_new_infections = self._manage_new_infections(recommender.user_profiles,
            recommender.infection_state)
        self.measurement_data.append(self.diffusion_tree.number_of_nodes())
        self._old_infection_state = np.copy(recommender.infection_state)

    def draw_tree(self):
        import matplotlib.pyplot as plt
        nx.draw(self.diffusion_tree, with_labels=True)


class StructuralVirality(DiffusionTreeMeasurement):
    import networkx as nx
    from networkx.algorithms.wiener import wiener_index
    def __init__(self, infection_state, verbose=False):
        DiffusionTreeMeasurement.__init__(self, infection_state, verbose)

    def get_structural_virality(self):
        n = self.diffusion_tree.number_of_nodes()
        return nx.wiener_index(self.diffusion_tree) / (n * (n - 1))
