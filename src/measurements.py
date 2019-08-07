import numpy as np

class Measurements():
    def __init__(self, num_items, num_users, debugger):
        self.delta_t = None
        self.index = None
        self.histogram_old = np.zeros(num_items)
        self.num_items = num_items
        self.num_users = num_users
        self.debugger = debugger.get_logger(__name__.upper())

    def _generate_interaction_histogram(self, interactions):
        histogram = np.zeros(self.num_items)
        np.add.at(histogram, interactions, 1)
        assert(histogram.sum() == self.num_users)
        return histogram

    # This measure of equilibrium corresponds to measuring whether interactions
    # are spread out among many items or only a few.
    # In other words, it looks at homogeneity vs heterogeneity
    def measure_equilibrium(self, interactions, step, interaction=None, visualization_rule=False):
        histogram = self._generate_interaction_histogram(interactions)
        histogram[::-1].sort()
        if self.debugger.is_enabled() and visualization_rule:
            self.debugger.pyplot_plot(np.arange(histogram.shape[0]), histogram,
                title="Sorted interaction histogram at step " + str(step),
                xlabel="Item", ylabel="# interactions")
        self.delta_t[self.index] = np.trapz(self.histogram_old, dx=1) - np.trapz(histogram, dx=1)
        self.histogram_old = np.copy(histogram)
        self.index += 1
        return histogram

    def get_delta(self):
        if self.debugger.can_show_results():
            self.debugger.pyplot_plot(np.arange(self.delta_t.shape[0]), self.delta_t,
                title='Heterogeneity', xlabel='Timestep', ylabel='Delta')
        return self.delta_t

    def set_delta(self, n):
        if self.delta_t is None:
            self.delta_t = np.empty(n)
            self.index = 0
        else:
            self.delta_t = np.resize(self.delta_t, len(self.delta_t) + n)