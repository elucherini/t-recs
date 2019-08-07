import numpy as np
import seaborn

import matplotlib.pyplot as plt

class UserPreferences():
    def __init__(self, num_users, num_items, debugger):
        self.debugger = debugger.get_logger(__name__.upper())
        self.preferences = self._compute_preferences(num_users, num_items)

    def _compute_preferences(self, num_users, num_items):
        #general_user_preferences = self._compute_general_preferences(num_users)
        #general_item_attributes = self._compute_general_attributes(num_items)
        #preferences = np.dot(general_user_preferences, general_item_attributes.T)
        #fraction = np.random.beta(np.full((num_users * num_items), 2), np.full((num_users * num_items), 10)).reshape((num_users, num_items))#self.distribution(**self.distribution_params)
        preferences = np.abs(np.random.normal(0, 5, size=(num_users, num_items)))
        if self.debugger.is_enabled():
            self.print_debug(preferences)
        # Only a fraction of the preferences is known to the users
        #return fraction * preferences
        return preferences

    def _compute_general_preferences(self, num_users):
        mu_rho = np.abs(np.random.normal(0, 5, size=(num_users, 1)))#10 * np.random.dirichlet(np.full((num_users), 1))
        #mu_rho[mu_rho < 0] = np.negative(mu_rho[mu_rho < 0])
        assert(mu_rho[mu_rho < 0].size == 0)
        return mu_rho#np.random.dirichlet(mu_rho).reshape((num_users, 1))

    def _compute_general_attributes(self, num_items):
        mu_alpha = np.random.random(size=(num_items, 1))
        # 0.1 * np.random.dirichlet(np.full((num_items), 100))
        return mu_alpha #np.random.dirichlet(mu_alpha).reshape((1, num_items))

    def get_preference_matrix(self, user=None):
        if user is None:
            return self.preferences
        else:
            return self.preferences[user, :]

    def get_user_choices(self, items, user_vector):
        m = self.preferences[user_vector.reshape((items.shape[0], 1)), items]
        self.debugger.log('User scores for given items are:\n' + str(m))
        return m.argsort()[:,::-1][:,0]

    def print_debug(self, preferences):
        best_items = preferences.argmax(axis=1)
        self.debugger.pyplot_plot(best_items, np.arange(preferences.shape[1]),
            plot_func=plt.hist, xlabel='Items', ylabel='# users who like item i the most',
            title='Histogram of users liking each item the most')
        plt.show()


