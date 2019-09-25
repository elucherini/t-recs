import numpy as np
from debug import Debug

'''
'' Class representing the measurement module
'''
class Measurements():
    '''
    '' @debugger: Debug instance
    '' @default_increment (optional): number of steps the measurement module is
    ''      initialized to. This is also the default increment in number of steps
    ''      when the measurement module runs out of free space.
    '''
    def __init__(self, debugger, default_increment = 20):
        self.delta_t = np.zeros(default_increment)
        self.mse = np.zeros(default_increment)
        self.index = 0
        self.histogram_old = None
        # Determine how many timesteps to set up each time
        self.default_increment = default_increment
        try:
            self.debugger = debugger.get_logger(__name__.upper())
        except AttributeError as e:
            print("Error! Measurements argument 'debugger' must be an instance of Debug")
            raise
        self.debugger.log("Measurement size set to: %d" % self.delta_t.size)

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
        self.debugger.log("Expanding measurement array size to: %d" % (array.size + timesteps))
        return np.resize(array, array.size + timesteps)

    def measure(self, step, interactions, num_users, num_items, predicted, actual,
        visualize=False):
        self._measure_equilibrium(step, interactions, num_users, num_items, visualize)
        self._measure_mse(predicted, actual, visualize)
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
    '' @visualize: if True, the module plots the sorted interaction histogram
    '''
    def _measure_equilibrium(self, step, interactions, num_users, num_items, 
        visualize=False):
        if self.delta_t is None or self.index >= self.delta_t.size:
            self.delta_t = self._expand_array(self.delta_t)
        assert(interactions.size == num_users)
        histogram = self._generate_interaction_histogram(interactions, num_users,
            num_items)
        histogram[::-1].sort()
        if self.histogram_old is None:
            self.histogram_old = np.zeros(num_items)
        if visualize:
            self.debugger.pyplot_plot(np.arange(histogram.shape[0]), histogram,
                title="Sorted interaction histogram at step " + str(step),
                xlabel="Item", ylabel="# interactions")
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

    def _measure_mse(self, predicted, actual, visualize=False):
        if self.mse is None or self.index >= self.mse.size:
            self.mse = self._expand_array(self.mse)
        self.mse[self.index] = ((predicted - actual)**2).mean()

    '''
    '' Return delta
    '''
    def _get_delta(self):
        #if self.debugger.can_show_results():
        collected_data = self.delta_t[:self.index]
        x = np.arange(collected_data.shape[0])

        return {'x': x, 'y': collected_data}

    '''
    '' Return mean squared error
    '''
    def _get_mse(self):
        collected_data = self.mse[:self.index]
        x = np.arange(collected_data.shape[0])
        return {'x': x, 'y': collected_data}

    '''
    '' Return all measurements
    '''
    def get_measurements(self):
        # TODO: generalize for all possible measures
        measurements = dict()
        measurements['delta'] = self._get_delta()
        measurements['mse'] = self._get_mse()

        return measurements

    def plot_measurements(self):
        measurements = self.get_measurements()
        ret = dict()
        for name, measure in measurements.items():
            self.debugger.pyplot_plot(measure['x'], measure['y'],
                title=str(name.capitalize()), xlabel='Timestep', 
                ylabel=str(name))
            ret[name] = measure
        return ret


if __name__ == '__main__':
    items = 10
    users = 5
    new_items = 2
    timesteps_one = 1
    timesteps_two = 2
    actual = np.random.randint(5, size=(users, items))
    predicted = actual + np.random.randint(-3,3, size=(users, items))

    debugger = Debug(__name__.upper(), False)
    logger = debugger.get_logger(__name__.upper())
    # Initialize
    meas = Measurements(debugger)

    # Interaction one
    interactions = np.random.randint(items, size=(1,users))
    logger.log('Add randomly generated interactions:\n%s' % str(interactions))
    meas.measure(step=0, interactions=interactions, num_users=users,
        num_items=items, predicted=predicted, actual=actual, visualize=True)

    # Interaction two
    interactions = np.random.randint(items, size=(1,users))
    logger.log('Add randomly generated interactions:\n%s' % str(interactions))
    meas.measure(step=1, interactions=interactions, num_users=users,
        num_items=items, predicted=predicted, actual=actual, visualize=True)

    # See graph
    m= meas.get_measurements()
    #logger.log('Delta: \n%s' % str(d))

    # Expand items
    logger.log('Expand with %d new items' % new_items)
    items += new_items

    # Interaction three
    interactions = np.random.randint(items, size=(1,users))
    logger.log('Add randomly generated interactions:\n%s' % str(interactions))
    meas.measure(step=2, interactions=interactions, num_users=users,
        num_items=items, predicted=predicted, actual=actual, visualize=True)

    # See graph
    m = meas.plot_measurements()
    logger.log('Delta: \n%s' % str(m))