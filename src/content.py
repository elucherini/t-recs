import numpy as np
import random
from abc import ABCMeta, abstractmethod
from recommender import Recommender
from measurements import Measurements
import matplotlib.pyplot as plt
import constants as const
from scipy.optimize import nnls

class ContentFiltering(Recommender):
    def __init__(self, num_users, num_items, items_representation=None, A=None, num_startup_iter=10,
        num_items_per_iter=10, randomize_recommended=True, num_recommended=None, num_new_items=None,
        user_preference=False, measurements=None):
        if items_representation is None:
            measurements = Measurements(num_items)
            self.beta_t = self._init_item_attributes(num_items, A)
        else:
            measurements = Measurements(items_representation.shape[0])
            self.beta_t = items_representation
        self.theta_t = self._init_user_profiles(num_users, A)#np.zeros((num_users, A))
        super().__init__(num_users, num_items, num_startup_iter, num_items_per_iter,
        randomize_recommended, num_recommended, num_new_items,
        user_preference, measurements)

    def _init_item_attributes(self, num_items, A):
        # TODO: non-random attributes?
        dist = abs(np.random.normal(0, 0.8, size=(A, num_items)).round(0))
        dist[np.where(dist > 1)] = 1
        # Is there any row with all item attributes set to zero?
        rows_not_zeros = np.any(dist > 0, axis=1)
        # If so, change a random element in the row(s) to one
        if False in rows_not_zeros:
            row = np.where(rows_not_zeros == False)[0]
            col = np.random.randint(0, A, size=(row.size))
            dist[row, col] = 1
        return dist

    def _init_user_profiles(self, num_users, A):
        # TODO: non-random attributes?
        dist = abs(np.random.normal(0, 0.8, size=(num_users, A)))
        dist[np.where(dist > 1)] = 1
        # Is there any row with all item attributes set to zero?
        rows_not_zeros = np.any(dist > 0, axis=1)
        # If so, change a random element in the row(s) to one
        if False in rows_not_zeros:
            row = np.where(rows_not_zeros == False)[0]
            col = np.random.randint(0, A, size=(row.size))
            dist[row, col] = 1
        return dist

    def _store_interaction(self, interactions):
        A = np.tile(self.beta_t, const.NUM_USERS)
        x, _ = nnls(A, interactions)

    def train(self):
        super().train()

    def recommend(self, k=1):
        super().recommend(k=k)

    def interact(self, user_vector=None, plot=False):
        if self.randomize_recommended:
            num_new_items = 1#np.random.randint(1, const.NUM_ITEMS_PER_ITER)#int(abs(np.random.normal(0, 2)) % const.NUM_ITEMS_PER_ITER)
            num_recommended = const.NUM_ITEMS_PER_ITER-num_new_items
        else:
            num_new_items = self.num_new_items
            num_recommended = self.num_recommended
        interactions = super().interact(user_vector, self.recommend(k=num_recommended), num_new_items)
        self._store_interaction(interactions)
        self.measure_equilibrium(interactions, plot=plot)

    def interact_startup(self):
        interactions = super().interact_startup(const.CONSTANT)
        self._store_interaction(interactions)
    
    def measure_equilibrium(self, interactions, plot=False):
        return self.measurements.measure_equilibrium(interactions, plot)


    def get_delta(self):
        return self.measurements.get_delta()
