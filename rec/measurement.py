from abc import ABCMeta, abstractmethod
from .debug import VerboseMode
import numpy as np

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
            self._old_histogram = np.zeros(num_items)
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
        """ Internal function that measures and records the mean
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
        print(((recommender.predicted_scores - recommender.actual_user_scores.actual_scores)))
        Measurement.measure(self, step, interactions, recommender)