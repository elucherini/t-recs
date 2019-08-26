import numpy as np
from debug import Debug

class Measurements():
    def __init__(self, num_items, num_users, debugger, default_increment = 20):
        self.delta_t = None
        self.index = 0
        self.histogram_old = np.zeros(num_items)
        self.num_items = num_items
        self.num_users = num_users
        # Determine how many timesteps to set up each time
        self.default_increment = default_increment
        self.debugger = debugger.get_logger(__name__.upper())
        self._set_delta()

    def _generate_interaction_histogram(self, interactions):
        histogram = np.zeros(self.num_items)
        np.add.at(histogram, interactions, 1)
        assert(histogram.sum() == self.num_users)
        return histogram

    def _set_delta(self, timesteps=None):
        if timesteps is None:
            timesteps = self.default_increment
        if self.delta_t is None:
            self.delta_t = np.zeros(timesteps)
            self.index = 0
        else:
            self.delta_t = np.resize(self.delta_t, len(self.delta_t) + timesteps)
        self.debugger.log("Delta size set to: %d" % self.delta_t.size)

    # This measure of equilibrium corresponds to measuring whether interactions
    # are spread out among many items or only a few.
    # In other words, it looks at homogeneity vs heterogeneity
    def measure_equilibrium(self, interactions, step, visualize=False):
        if self.delta_t is None or self.index >= self.delta_t.size:
            self._set_delta()
        histogram = self._generate_interaction_histogram(interactions)
        histogram[::-1].sort()
        if visualize:
            self.debugger.pyplot_plot(np.arange(histogram.shape[0]), histogram,
                title="Sorted interaction histogram at step " + str(step),
                xlabel="Item", ylabel="# interactions")
        self.delta_t[self.index] = np.trapz(self.histogram_old, dx=1) - \
                                    np.trapz(histogram, dx=1)
        self.histogram_old = np.copy(histogram)
        self.index += 1
        return histogram

    def expand_items(self, num_new_items):
        self.num_items += num_new_items

    def get_delta(self):
        if self.debugger.can_show_results():
            x = np.arange(self.delta_t[:self.index].shape[0])
            y = self.delta_t[:self.index]
            self.debugger.pyplot_plot(x, y, title='Heterogeneity', xlabel='Timestep', 
                ylabel='Delta')
        return self.delta_t[:self.index]

if __name__ == '__main__':
    num_items = 10
    num_users = 5
    num_new_items = 2
    timesteps_one = 1
    timesteps_two = 2

    debugger = Debug(__name__.upper(), False)
    logger = debugger.get_logger(__name__.upper())
    # Initialize
    meas = Measurements(num_items, num_users, debugger)

    # Interaction one
    interactions = np.random.randint(num_items, size=(1,num_users))
    logger.log('Add randomly generated interactions:\n%s' % str(interactions))
    meas.measure_equilibrium(interactions, step=0, visualize=True)

    # Interaction two
    interactions = np.random.randint(num_items, size=(1,num_users))
    logger.log('Add randomly generated interactions:\n%s' % str(interactions))
    meas.measure_equilibrium(interactions, step=1, visualize=True)

    # See graph
    d= meas.get_delta()
    logger.log('Delta: \n%s' % str(d))

    # Expand items
    logger.log('Expand with %d new items' % num_new_items)
    meas.expand_items(num_new_items)
    num_items += num_new_items

    # Interaction three
    interactions = np.random.randint(num_items, size=(1,num_users))
    logger.log('Add randomly generated interactions:\n%s' % str(interactions))
    meas.measure_equilibrium(interactions, step=2, visualize=True)

    # See graph
    d = meas.get_delta()
    logger.log('Delta: \n%s' % str(d))