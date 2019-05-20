import numpy as np

class UserPreferences():
    def __init__(self, num_users, num_items, num_items):
        self.ranked_items = None
        self.f = lambda n, c: n**(-abs(c))
        self.P = None #self._compute_p(num_users, num_items)
        self.rho = self._compute_rho(num_users)
        self.alpha = self._compute_alpha(num_items)

    def _compute_general_preferences(self, num_users):
        mu_rho = 10 * np.random.dirichlet(np.full((num_users), 1))
        return np.random.dirichlet(mu_rho).reshape((num_users, 1))

    def _compute_general_attributes(self, num_items):
        mu_alpha = 0.1 * np.random.dirichlet(np.full((num_items), 100))
        return np.random.dirichlet(mu_alpha).reshape((1, num_items))

    def _compute_p(self, num_users, num_items):
        product = np.dot(self.rho, self.alpha)
        # FIXME look into scipy.stats too
        #V = np.random.beta(product.reshape(-1,))
        #return np.random.beta(mu_eta = 0.98) * V

    def get_general_preferences(self):
        return self.rho

    def get_general_attributes(self):
        return self.alpha