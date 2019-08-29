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
        self.index = 0
        self.histogram_old = None
        # Determine how many timesteps to set up each time
        self.default_increment = default_increment
        try:
            self.debugger = debugger.get_logger(__name__.upper())
        except Exception as e:
            print(e)
        self.debugger.log("Delta size set to: %d" % self.delta_t.size)
        #self._expand_delta()

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
    '' @timesteps: number of steps to add 
    '''
    def _expand_delta(self, timesteps=None):
        if timesteps is None:
            timesteps = self.default_increment
        self.delta_t = np.resize(self.delta_t, len(self.delta_t) + timesteps)
        self.debugger.log("Delta size expanded to: %d" % self.delta_t.size)

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
    def measure_equilibrium(self, step, interactions, num_users, num_items, 
        visualize=False):
        if self.delta_t is None or self.index >= self.delta_t.size:
            self._expand_delta()
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
        self.index += 1
        return histogram

    '''
    '' Return measurement
    '''
    def get_delta(self):
        if self.debugger.can_show_results():
            x = np.arange(self.delta_t[:self.index].shape[0])
            y = self.delta_t[:self.index]
            self.debugger.pyplot_plot(x, y, title='Heterogeneity', xlabel='Timestep', 
                ylabel='Delta')
        return self.delta_t[:self.index]

if __name__ == '__main__':
    items = 10
    users = 5
    new_items = 2
    timesteps_one = 1
    timesteps_two = 2

    debugger = Debug(__name__.upper(), False)
    logger = debugger.get_logger(__name__.upper())
    # Initialize
    meas = Measurements(debugger)

    # Interaction one
    interactions = np.random.randint(items, size=(1,users))
    logger.log('Add randomly generated interactions:\n%s' % str(interactions))
    meas.measure_equilibrium(step=0, interactions=interactions, num_users=users,
        num_items=items, visualize=True)

    # Interaction two
    interactions = np.random.randint(items, size=(1,users))
    logger.log('Add randomly generated interactions:\n%s' % str(interactions))
    meas.measure_equilibrium(step=1, interactions=interactions, num_users=users,
        num_items=items, visualize=True)

    # See graph
    d= meas.get_delta()
    logger.log('Delta: \n%s' % str(d))

    # Expand items
    logger.log('Expand with %d new items' % new_items)
    items += new_items

    # Interaction three
    interactions = np.random.randint(items, size=(1,users))
    logger.log('Add randomly generated interactions:\n%s' % str(interactions))
    meas.measure_equilibrium(step=2, interactions=interactions, num_users=users,
        num_items=items, visualize=True)

    # See graph
    d = meas.get_delta()
    logger.log('Delta: \n%s' % str(d))