import numpy as np
import random
from abc import ABCMeta, abstractmethod
from recommender import Recommender
from measurements import Measurements
import constants as const

class PopularityRecommender(Recommender):
    def __init__(self, num_users, num_items, num_startup_iter=10, num_items_per_iter=10,
        randomize_recommended=True, num_recommended=None, num_new_items=None,
        user_preference=False):
        # TODO: check on invalid parameters
        self.theta_t = np.ones((num_users, 1), dtype=int)
        self.beta_t = np.zeros((1, num_items), dtype=int)
        super().__init__(num_users, num_items, num_startup_iter, num_items_per_iter,
        randomize_recommended, num_recommended, num_new_items,
        user_preference, Measurements(num_items))

    def _store_interaction(self, interactions):
        self.beta_t = np.add(self.beta_t, interactions)

    def train(self):
        return super().train()

    def measure_equilibrium(self, interactions, plot=False):
        return self.measurements.measure_equilibrium(interactions, plot)

    def interact(self, user_vector=None, plot=False, startup=False):
        if startup:
            num_new_items = self.num_items_per_iter
            num_recommended = 0
        elif self.randomize_recommended:
            num_new_items = np.random.randint(1, self.num_items_per_iter)
            num_recommended = self.num_items_per_iter-num_new_items
        else:
            num_new_items = self.num_new_items
            num_recommended = self.num_recommended
        recommended = self.recommend(k=num_recommended) if not startup else None
        interactions = super().interact(user_vector, recommended, num_new_items)
        self._store_interaction(interactions)
        self.measure_equilibrium(interactions, plot=plot)

    # Recommends items proportional to their popularity
    def recommend(self, k=1):
        rec = self.s_t.argsort()
        p = np.arange(1, self.num_items + 1)
        return np.random.choice(rec[0], p=p/p.sum(), size=(self.num_users, k))

    def get_delta(self):
        return self.measurements.get_delta()