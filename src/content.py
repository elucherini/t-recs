import numpy as np
from recommender import Recommender
from measurements import Measurements
from user_preferences import UserPreferences
import matplotlib.pyplot as plt
from scipy.optimize import nnls

class ContentFiltering(Recommender):
    def __init__(self, num_users, num_items, items_representation=None, A=100, num_startup_iter=10,
        num_items_per_iter=10, randomize_recommended=True, num_recommended=None, num_new_items=None,
        user_preferences=False, debug_user_preferences=False, measurements=None):
        if items_representation is None:
            measurements = Measurements(num_items, num_users)
            self.item_attributes = self._init_random_item_attributes(A, num_items)
        else:
            self.item_attributes = items_representation
            measurements = Measurements(self.item_attributes.shape[1], num_users)
        # TODO: user profiles should be learned from users' interactions
        self.user_profiles = np.zeros((num_users, A))#self._init_user_profiles(A, num_users)
        if user_preferences:
            preferences = UserPreferences(num_users, num_items, debug=debug_user_preferences)
        else:
            preferences = None
        super().__init__(num_users, num_items, num_startup_iter, num_items_per_iter,
            randomize_recommended, num_recommended, num_new_items,
            preferences, measurements)
        #print(self.item_attributes.shape)

    def _init_random_item_attributes(self, A, num_items):
        # Binary representations
        # TODO: non-random attributes?
        dist = abs(np.random.normal(0, 0.8, size=(num_items, A)).round(0))
        dist[np.where(dist > 1)] = 1
        # Is there any row with all item attributes set to zero?
        rows_not_zeros = np.any(dist > 0, axis=1)
        # If so, change a random element in the row(s) to one
        if False in rows_not_zeros:
            row = np.where(rows_not_zeros == False)[0]
            col = np.random.randint(0, A, size=(row.size))
            dist[row, col] = 1
        return dist.T

    def _init_user_profiles(self, A, num_users):
        # Real numbers
        # TODO: non-random attributes?
        dist = abs(np.random.normal(0, 0.8, size=(num_users, A)))
        # Is there any row with all item attributes set to zero?
        rows_not_zeros = np.any(dist > 0, axis=1)
        # If so, change a random element in the row(s)
        if False in rows_not_zeros:
            row = np.where(rows_not_zeros == False)[0]
            col = np.random.randint(0, A, size=(row.size))
            dist[row, col] = np.random.rand()
        return dist

    def _store_interaction(self, interactions):
        interactions_per_user = np.zeros((self.num_users, self.num_items))
        interactions_per_user[self.user_vector, interactions] = 1
        user_attributes = np.dot(interactions_per_user, self.item_attributes.T)
        self.user_profiles = np.add(self.user_profiles, user_attributes)
        #x, _ = nnls(A, interactions)

    def train(self):
        # Normalize user_profiles
        user_profiles = self.user_profiles / self.user_profiles.sum()
        super().train(user_profiles=user_profiles)
        #print(self.scores)

    def interact(self, plot=False, step=None, startup=False):
        if startup:
            num_new_items = self.num_items_per_iter
            num_recommended = 0
        elif self.randomize_recommended:
            num_new_items = np.random.randint(1, self.num_items_per_iter)
            num_recommended = self.num_items_per_iter-num_new_items
        else:
            # TODO: these may be constants or iterators on vectors
            num_new_items = self.num_new_items
            num_recommended = self.num_recommended
        recommended = self.recommend(k=num_recommended) if not startup else None
        if num_recommended > 0:
            assert(num_recommended == recommended.shape[1])
        interactions = super().interact(recommended, num_new_items)
        self.measure_equilibrium(interactions, plot=plot, step=step)
        self._store_interaction(interactions)

    def recommend(self, k=1):
        return super().recommend(k=k)

    def startup_and_train(self, timesteps=50, debug=False):
        return super().run(timesteps, startup=True, train=False, debug=debug)

    def run(self, timesteps=50, train=True, debug=False):
        return super().run(timesteps, startup=False, train=train, debug=debug)

    def get_heterogeneity(self):
        return self.measurements.get_delta()

    def measure_equilibrium(self, interactions, plot=False, step=None):
        return self.measurements.measure_equilibrium(interactions, plot=plot, step=step)