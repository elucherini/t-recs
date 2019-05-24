import numpy as np

import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

class Measurements():
    def __init__(self, num_items):
        self.delta_t = None
        self.index = None
        self.interactions_old = np.zeros(num_items)

    # This measure of equilibrium corresponds to measuring whether popularity is spread out among many items or only a few.
    # In other words, it looks at homogeneity vs heterogeneity
    def measure_equilibrium(self, interactions, plot=False):
        interactions[::-1].sort()
        if plot:
            plt.plot(np.arange(len(interactions)), interactions)
            plt.show()
        self.delta_t[self.index] = np.trapz(self.interactions_old, dx=1) - np.trapz(interactions, dx=1)
        self.interactions_old = np.copy(interactions)
        self.index += 1

    def get_delta(self):
        return self.delta_t

    def set_delta(self, n):
        if self.delta_t is None:
            self.delta_t = np.empty(n)
            self.index = 0
        else:
            self.delta_t = np.resize(self.delta_t, len(self.delta_t) + n)