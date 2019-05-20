import numpy as np
import random
from abc import ABCMeta, abstractmethod
from recommender import Recommender
from measurements import Measurements
import matplotlib.pyplot as plt
import constants as const

class ContentFiltering(Recommender):
    def __init__(self, num_users, items_representation=None, num_items=None, A=None):
        if items_representation is None:
            self.measurements = Measurements(num_items)
            self.beta_t = self._init_item_attributes(num_items, A)
        else:
            self.measurements = Measurements(items_representation.shape[0])
            self.beta_t = items_representation
        self.theta_t = np.zeros((num_users, A))
        self.s_t = None
        self.new_items_iter = None
        #self.num_users = num_users
        #self.num_items = num_items

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

    def train(self):
        super().train()

    def recommend(self, k=1):
        super().recommend(k=k)

    def interact(self):
        pass

    def interact_startup(self):
        pass
    
    def measure_equilibrium(self, interactions):
        pass

if __name__ == '__main__':
    # A = number of attribute tags, determines dimensions of theta_t and beta_t
    A = 100
    rec = ContentFiltering(num_users=const.NUM_USERS, num_items=const.NUM_ITEMS, A=A)

    # Startup
    rec.interact_startup(const.NUM_STARTUP_ITER, num_items_per_iter=const.NUM_ITEMS_PER_ITER, random_preference=True)
    rec.train()

    users = np.arange(const.NUM_USERS, dtype=int)

    # Runtime
    for t in range(const.TIMESTEPS - const.NUM_ITEMS_PER_ITER):
        rec.recommend(k=2)
        rec.interact(user_vector=users, num_recommended=1, 
            num_new_items=const.NUM_ITEMS_PER_ITER - 1, random_preference=True)
        rec.train()

    delta_t = rec.get_delta()
    plt.plot(np.arange(len(delta_t)), delta_t)
    plt.show()