import numpy as np
from .debug import VerboseMode
from .utils import toDataFrame

class Measurements(VerboseMode):
    '''Class representing the measurement module.
    
        Args:
            default_increment (int, optional): number of steps the measurement module is
                initialized to. This is also the default increment in number of steps
                when the measurement module runs out of free space.
            verbose (bool, optional): If True, enables verbose mode. Disabled by default.

        Attributes:
            Attributes inherited by :class:`VerboseMode`, plus:
            delta_t (:obj:`numpy.array`): An array containing a measurement of
                heterogeneity per timestep.
            mse (:obj:`numpy.array`): An array containing the mean squared error
                at each timestep.

        Private attributes:
            _index (int): the index of the first free position in the arrays. This
                implementation expects that all arrays are updated at each timestep.
            _histogram_old (:obj:`numpy.array` or None): contains the previous histogram
                of user interactions. It is used by the methods that measure homogeneity.
            _default_increment (int): default number of positions the arrays of the measurement
                module are icremented by when they run out of free space.
    '''
    def __init__(self, default_increment = 20, verbose=False):
        self.delta_t = np.zeros(default_increment)
        self.mse = np.zeros(default_increment)
        self._index = 0
        self._histogram_old = None
        # Determine how many timesteps to set up each time
        self._default_increment = default_increment
        super().__init__(__name__.upper(), verbose)
        self.log("Measurement size set to: %d" % self.delta_t.size)

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

    def _expand_array(self, array, timesteps=None):    
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
        self.log("Expanding measurement array size to: %d" % (array.size + timesteps))
        return np.resize(array, array.size + timesteps)

    def measure(self, step, interactions, num_users, num_items, predicted, actual):
        """ Executes and records all measurements in the module.
            
            Args:
                step (int): current time step
                interactions (:obj:`numpy.array`): array of interactions per user.
                num_users (int): number of users in the system
                num_items (int): number of items in the system
                predicted (:obj:`numpy.array`): user preferences predicted by
                    the system.
                actual (:obj:`numpy.array`): actual user preferences.
        """
        self._measure_equilibrium(step, interactions, num_users, num_items)
        self._measure_mse(predicted, actual)
        self._index += 1

    def _measure_equilibrium(self, step, interactions, num_users, num_items):   
        '''
            Internal function that measures the homogeneity of user interactions
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
        if self.delta_t is None or self._index >= self.delta_t.size:
            self.delta_t = self._expand_array(self.delta_t)
        assert(interactions.size == num_users)
        histogram = self._generate_interaction_histogram(interactions, num_users,
            num_items)
        histogram[::-1].sort()
        if self._histogram_old is None:
            self._histogram_old = np.zeros(num_items)
        # delta(t) = Area(histogram(t-1)) - Area(histogram(t))
        self.delta_t[self._index] = np.trapz(self._histogram_old, dx=1) - \
                                    np.trapz(histogram, dx=1)
        self._histogram_old = np.copy(histogram)
        return histogram

    def _measure_mse(self, predicted, actual):
        """ Internal function that measures and records the mean
            squared error between the user preferences predicted
            by the system and the users' actual preferences.

            Args:
                predicted (:obj:`numpy.array`): user preferences
                    predicted by the system
                actual (:obj:`numpy.array`): actual user preferences,
                    unknown to the system.
        """
        if self.mse is None or self._index >= self.mse.size:
            self.mse = self._expand_array(self.mse)
        #print(np.where(predicted == 0)[0].shape)
        #print(np.where(actual == 0)[0].shape)

        self.mse[self._index] = ((predicted - actual)**2).mean()
        #print((predicted - actual)**2)

    def _get_delta(self):
        """ Returns a measure of heterogeneity of content.

            Returns: dict with measure of homogeneity per timestep.
        """
        collected_data = self.delta_t[:self._index]
        x = np.arange(collected_data.shape[0])

        return {'Timestep': x, 'Homogeneity': collected_data}

    def _get_mse(self):
        """ Returns mean squared error between the user preferences 
            predicted by the system and the users' actual preferences.

            Returns: dict of mean squared error per timestep.

        """
        collected_data = self.mse[:self._index]
        x = np.arange(collected_data.shape[0])
        return {'Timestep': x, 'MSE': collected_data}

    def get_measurements(self):
        """ Returns all measurements recorded by the module.

            Returns: dict with all measurements per timestep.
        """
        # TODO: generalize for all possible measures
        measurements = dict()
        measurements['delta'] = self._get_delta()
        measurements['mse'] = self._get_mse()
        data = {**measurements['delta'], **measurements['mse']}
        return data
