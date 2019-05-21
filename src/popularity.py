import numpy as np
import random
from abc import ABCMeta, abstractmethod
from recommender import Recommender
from measurements import Measurements
import matplotlib.pyplot as plt
import constants as const

plt.style.use('seaborn-whitegrid')

class PopularityRecommender(Recommender):
    def __init__(self, num_users, num_items):
        self.theta_t = np.ones((num_users, 1), dtype=int)
        self.beta_t = np.zeros((1, num_items), dtype=int)
        self.s_t = None
        self.measurements = Measurements(num_items)
        self.new_items_iter = None
        self.num_users = num_users
        self.num_items = num_items

    def _store_interaction(self, interactions):
        self.beta_t = np.add(self.beta_t, interactions)

    def train(self):
        return super().train()

    def measure_equilibrium(self, interactions):
        return self.measurements.measure_equilibrium(interactions)

    def interact_startup(self, num_startup_iter, num_items_per_iter=10, random_preference=True,
                                                                preference=None):
        interactions = super().interact_startup(num_startup_iter, num_items_per_iter, 
            random_preference, preference, const.CONSTANT)
        self._store_interaction(interactions)

    def interact(self, user_vector=None, num_recommended=5, num_new_items=5, random_preference=True,
                                                                                    preference=None):
        interactions = super().interact(user_vector, num_recommended, num_new_items, 
            random_preference, preference, self.recommend(k=num_recommended))
        self._store_interaction(interactions)
        self.measure_equilibrium(interactions)

    def recommend(self, k=1):
        return super().recommend(k=k)

    def get_delta(self):
        return self.measurements.get_delta()


if __name__ == '__main__':
    rec = PopularityRecommender(const.NUM_USERS, const.NUM_ITEMS)
    # Startup
    rec.interact_startup(const.NUM_STARTUP_ITER, num_items_per_iter=const.NUM_ITEMS_PER_ITER, 
        random_preference=True)
    rec.train()

    users = np.arange(const.NUM_USERS, dtype=int)

    # Runtime
    for t in range(const.TIMESTEPS - const.NUM_ITEMS_PER_ITER):
        rec.interact(user_vector=users, num_recommended=1, num_new_items=const.NUM_ITEMS_PER_ITER - 1, 
            random_preference=True)
        rec.train()

    delta_t = rec.get_delta()
    plt.plot(np.arange(len(delta_t)), delta_t)
    plt.show()