import numpy as np

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

class Measurements():
    def __init__(self, num_items, num_users):
        self.delta_t = None
        self.index = None
        self.histogram_old = np.zeros(num_items)
        self.num_items = num_items
        self.num_users = num_users

    def _generate_interaction_histogram(self, interactions):
        histogram = np.zeros(self.num_items)
        np.add.at(histogram, interactions, 1)
        assert(histogram.sum() == self.num_users)
        return histogram

    # This measure of equilibrium corresponds to measuring whether interactions
    # are spread out among many items or only a few.
    # In other words, it looks at homogeneity vs heterogeneity
    def measure_equilibrium(self, interactions, plot=False, step=None, interaction=None):
        histogram = self._generate_interaction_histogram(interactions)
        histogram[::-1].sort()
        if plot:
            plot_b = 1
            plt.plot(np.arange(len(histogram)), histogram)
            plt.title("Sorted interaction histogram at step " + str(step))
            plt.xlabel("Item")
            plt.ylabel("# interactions")
            plt.show()
        self.delta_t[self.index] = np.trapz(self.histogram_old, dx=1) - np.trapz(histogram, dx=1)
        self.histogram_old = np.copy(histogram)
        self.index += 1
        return histogram

    def get_delta(self):
        return self.delta_t

    def set_delta(self, n):
        if self.delta_t is None:
            self.delta_t = np.empty(n)
            self.index = 0
        else:
            self.delta_t = np.resize(self.delta_t, len(self.delta_t) + n)