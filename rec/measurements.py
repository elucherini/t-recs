import numpy as np
from .debug import VerboseMode
from .utils import toDataFrame

'''
'' Class representing the measurement module
'''
class Measurements(VerboseMode):
    '''
    '' @debugger: Debug instance
    '' @default_increment (optional): number of steps the measurement module is
    ''      initialized to. This is also the default increment in number of steps
    ''      when the measurement module runs out of free space.
    '''
    def __init__(self, default_increment = 20, verbose=False):
        self.delta_t = np.zeros(default_increment)
        self.mse = np.zeros(default_increment)
        self.index = 0
        self.histogram_old = None
        # Determine how many timesteps to set up each time
        self.default_increment = default_increment
        super().__init__(__name__.upper(), verbose)
        self.log("Measurement size set to: %d" % self.delta_t.size)

    '''
    '' Internal function that returns a histogram of the number 
    '' of interactions per item at the given timestep.
    '' @interactions: non-aggregated array of interactions
    '' @num_users: number of users in the system
    '' @num_items: number of items in the system
    '''
    def _generate_interaction_histogram(self, interactions, num_users, num_items):
        histogram = np.zeros(num_items)
        np.add.at(histogram, interactions, 1)
        # Check that there's one interaction per user
        assert(histogram.sum() == num_users)
        return histogram

    '''
    '' Internal function to expand measurement module to accommodate 
    '' more time steps.
    '' @array: measurement array to expand
    '' @timesteps: number of steps to add 
    '''
    def _expand_array(self, array, timesteps=None):
        if timesteps is None:
            timesteps = self.default_increment
        self.log("Expanding measurement array size to: %d" % (array.size + timesteps))
        return np.resize(array, array.size + timesteps)

    def measure(self, step, interactions, num_users, num_items, predicted, actual):
        self._measure_equilibrium(step, interactions, num_users, num_items)
        self._measure_mse(predicted, actual)
        self.index += 1

    '''
    '' Measure of the homogeneity of interactions (i.e., whether interactions
    '' are widespread among many items or only a few items)
    '' @step: current time step
    '' @interactions: non-aggregated array of interactions (i.e.,
    ''      array of length |U| s.t. element u is the index of
    ''      item user u interacted with)
    '' @num_users: number of users in the system
    '' @num_items: number of items in the system
    '''
    def _measure_equilibrium(self, step, interactions, num_users, num_items):
        if self.delta_t is None or self.index >= self.delta_t.size:
            self.delta_t = self._expand_array(self.delta_t)
        assert(interactions.size == num_users)
        histogram = self._generate_interaction_histogram(interactions, num_users,
            num_items)
        histogram[::-1].sort()
        if self.histogram_old is None:
            self.histogram_old = np.zeros(num_items)
        # delta(t) = Area(histogram(t-1)) - Area(histogram(t))
        self.delta_t[self.index] = np.trapz(self.histogram_old, dx=1) - \
                                    np.trapz(histogram, dx=1)
        self.histogram_old = np.copy(histogram)
        return histogram

    '''
    '' Measure mean squared error
    '' @predicted:
    '' @actual:
    '''
    def _measure_mse(self, predicted, actual):
        if self.mse is None or self.index >= self.mse.size:
            self.mse = self._expand_array(self.mse)
        self.mse[self.index] = ((predicted - actual)**2).mean()

    '''
    '' Return delta
    '''
    def _get_delta(self):
        #if self.verbose.can_show_results():
        collected_data = self.delta_t[:self.index]
        x = np.arange(collected_data.shape[0])

        return {'Timestep': x, 'Homogeneity': collected_data}

    '''
    '' Return mean squared error
    '''
    def _get_mse(self):
        collected_data = self.mse[:self.index]
        x = np.arange(collected_data.shape[0])
        return {'Timestep': x, 'MSE': collected_data}

    '''
    '' Return all measurements
    '''
    def get_measurements(self):
        # TODO: generalize for all possible measures
        measurements = dict()
        measurements['delta'] = self._get_delta()
        measurements['mse'] = self._get_mse()
        data = {**measurements['delta'], **measurements['mse']}
        return toDataFrame(data, index='Timestep')
