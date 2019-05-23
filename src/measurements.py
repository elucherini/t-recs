import numpy as np

import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

class Measurements():
    def __init__(self, num_items):
        self.delta_t = list()
        self.interactions_old = np.zeros(num_items)

    # This measure of equilibrium corresponds to measuring whether popularity is spread out among many items or only a few.
    # In other words, it looks at homogeneity vs heterogeneity
    def measure_equilibrium(self, interactions, plot=False):
        interactions[::-1].sort()
        if plot:
            plt.plot(np.arange(len(interactions)), interactions)
            plt.show()
        self.delta_t.append(np.trapz(self.interactions_old, dx=1) - np.trapz(interactions, dx=1))
        self.interactions_old = np.copy(interactions)

    def get_delta(self):
        return self.delta_t