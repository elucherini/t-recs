import numpy as np
import seaborn

import matplotlib.pyplot as plt
from debug import Debug

class ActualUserScores():
    def __init__(self, num_users, item_representation, debugger):
        self.debugger = debugger.get_logger(__name__.upper())
        self.actual_scores = self._compute_actual_scores(num_users, item_representation,
            spread=20)

    def _compute_actual_scores(self, num_users, item_representation, 
        spread=1000):
        user_profiles = abs(np.random.normal(0, spread, 
            size=(num_users, item_representation.shape[0])))
        user_profiles = user_profiles / user_profiles.sum(axis=1)[:,None]
        # Calculate actual user scores
        actual_scores = np.dot(user_profiles, item_representation)
        self.debugger.log("Items:\n" + str(item_representation))
        self.debugger.log("User profiles:\n" + str(user_profiles))
        self.debugger.log("Actual score:\n" + str(actual_scores))
        if self.debugger.is_enabled():
            self.print_debug(actual_scores)
        return actual_scores

    def _compute_general_preferences(self, num_users):
        mu_rho = np.abs(np.random.normal(0, 5, size=(num_users, 1)))
        #10 * np.random.dirichlet(np.full((num_users), 1))
        #mu_rho[mu_rho < 0] = np.negative(mu_rho[mu_rho < 0])
        assert(mu_rho[mu_rho < 0].size == 0)
        return mu_rho#np.random.dirichlet(mu_rho).reshape((num_users, 1))

    def _compute_general_attributes(self, num_items):
        mu_alpha = np.random.random(size=(num_items, 1))
        # 0.1 * np.random.dirichlet(np.full((num_items), 100))
        return mu_alpha #np.random.dirichlet(mu_alpha).reshape((1, num_items))

    def get_actual_user_scores(self, user=None):
        if user is None:
            return self.actual_scores
        else:
            return self.actual_scores[user, :]

    def get_user_choices(self, items, user_vector):
        m = self.actual_scores[user_vector.reshape((items.shape[0], 1)), items]
        self.debugger.log('User scores for given items are:\n' + str(m))
        return m.argsort()[:,::-1][:,0]

    def print_debug(self, actual_scores):
        best_items = actual_scores.argmax(axis=1)
        self.debugger.log('Shape: ' + str(actual_scores.shape))
        self.debugger.pyplot_plot(best_items, np.arange(actual_scores.shape[1] + 1),
            plot_func=plt.hist, xlabel='Items', ylabel='# users who like item i the most',
            title='Histogram of users liking each item the most')
        plt.show()

# Unit test
if __name__ == '__main__':
    # Debugger module
    debugger = Debug(__name__.upper(), True)

    num_users = 1000
    num_items = 1125
    A = num_items


    # Random normalized representation
    #item_representation = np.random.randint(0, num_items, size=(num_items, A))
    #item_representation = item_representation / item_representation.sum(axis=1)[:,None]

    # Random binary item representation
    item_representation = np.random.binomial(1, .3, size=(num_items, A))
    actual_scores = ActualUserScores(num_users, item_representation.T, debugger)
