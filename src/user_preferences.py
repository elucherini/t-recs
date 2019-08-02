import numpy as np
import seaborn

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

class UserPreferences():
    def __init__(self, num_users, num_items, debug=False):
        self.preferences = self._compute_preferences(num_users, num_items)
        self.debug = debug
        if self.debug:
            self.print_debug()

    def _compute_preferences(self, num_users, num_items):
        general_user_preferences = self._compute_general_preferences(num_users)
        general_item_attributes = self._compute_general_attributes(num_items)
        #print(np.nonzero(general_user_preferences)[0].shape)
        #print(np.nonzero(general_item_attributes)[0].shape)
        preferences = np.dot(general_user_preferences, general_item_attributes)
        fraction = np.random.beta(np.full((num_users * num_items), 2), np.full((num_users * num_items), 10)).reshape((num_users, num_items))#self.distribution(**self.distribution_params)
        #print(np.nonzero(preferences)[0].shape)
        #print(np.nonzero(fraction)[0].shape)
        # Only a fraction of the preferences is known to the users
        return fraction * preferences

    def _compute_general_preferences(self, num_users):
        mu_rho = np.random.normal(0, 5, size=(num_users, 1))#10 * np.random.dirichlet(np.full((num_users), 1))
        mu_rho[mu_rho < 0] = np.negative(mu_rho[mu_rho < 0])
        assert(mu_rho[mu_rho < 0].size == 0)
        return mu_rho#np.random.dirichlet(mu_rho).reshape((num_users, 1))

    def _compute_general_attributes(self, num_items):
        mu_alpha = np.floor(np.random.normal(1, 2, size=(1, num_items)))
        mu_alpha[mu_alpha > 1] = 1
        mu_alpha[mu_alpha < 0] = 0
        assert(mu_alpha[mu_alpha > 1].size == 0)
        assert(mu_alpha[mu_alpha < 0].size == 0)
        # 0.1 * np.random.dirichlet(np.full((num_items), 100))
        return mu_alpha #np.random.dirichlet(mu_alpha).reshape((1, num_items))

    def get_user_choices(self, items, user_vector):
        m = self.preferences[user_vector.reshape((items.shape[0], 1)), items]
        return m.argsort()[:,::-1][:,0]

    def print_debug(self):
        # Plot heatmap of user preference for each item
        seaborn.heatmap(self.preferences).set(xlabel='Items', ylabel='Users', title="User preference for each item")
        plt.show()
        best_items = self.preferences.argmax(axis=1)
        unique, counts = np.unique(best_items, return_counts=True)
        plt.hist(counts, bins=unique)
        plt.xlabel('Items')
        plt.ylabel('# users who like item i the most')
        plt.title('Histogram of users liking each item the most')
        plt.show()
