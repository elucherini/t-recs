import numpy as np
import random
from abc import ABCMeta, abstractmethod
from recommender import Recommender
from measurements import Measurements
import matplotlib.pyplot as plt
import constants as const
import timeit

plt.style.use('seaborn-whitegrid')

class PopularityRecommender(Recommender):
    def __init__(self, num_users, num_items):
        self.measurements = Measurements(num_items)
        self.theta_t = np.ones((num_users, 1), dtype=int)
        self.beta_t = np.zeros((1, num_items), dtype=int)
        self.s_t = None
        self.new_items_iter = None
        self.num_users = num_users
        self.num_items = num_items

    # Stores interaction without training
    def _store_interaction(self, interactions):
        self.beta_t = np.add(self.beta_t, interactions)

    # Trains model; it either adds new interactions,
    # or it updates the score with the stored interactions
    def train(self, interactions=None):
        if interactions is not None:
            self.beta_t = np.add(self.beta_t, interactions)
        self.s_t = np.dot(self.theta_t, self.beta_t)

    # Return matrix that can be stored or used to train model
    def _generate_interaction_matrix(self, interactions):
        tot_interactions = np.zeros(self.num_items)
        np.add.at(tot_interactions, interactions, 1)
        return tot_interactions

    def measure_equilibrium(self, interactions):
        return self.measurements.measure_equilibrium(interactions)

    def interact_startup(self, num_startup_iter, num_items_per_iter=10, random_preference=True, preference=None):
        # Current assumptions:
        # 1. First (num_startup_iter * num_items_per_iter) items presented for startup
        # 2. New  num_items_per_iter items at each interaction, no recommendations
        # 3. TODO: consider user preferences that are not random
        # TODO: remove loop for time here or optimize
        self.new_items_iter = iter(np.arange(num_startup_iter * num_items_per_iter, self.num_items).reshape(-1, num_items_per_iter))
        if random_preference is True:
            preference = np.zeros(self.num_users * num_startup_iter, dtype=int)
        index = 0
        for t in np.arange(1, num_startup_iter + 1):
            if random_preference is True:
                preference[index:index+self.num_users] = np.random.randint((t-1) * num_items_per_iter, t * num_items_per_iter, size=(self.num_users))
            index += self.num_users
        interactions = self._generate_interaction_matrix(preference)
        self._store_interaction(interactions)

    #def generate_interactions(self, num_iter, num_items_per_iter=10, num_new_items=5, random_preference=True, preference=None):
    def interact(self, num_recommended=5, num_new_items=5, random_preference=True, preference=None):
        # Current assumptions:
        # 1. Interleave new items and recommended items
        # 2. Fixed number of new/recommended items
        # 3. New items are chosen randomly from the same set of num_items_per_iter items
        # 4. Each user interacts with different elements depending on preference
        # 5. Train after every interaction
        # TODO: each user can interact with each element at most once
        # TODO: consider user preferences that are not random
        num_items_per_iter = num_new_items + num_recommended
        #interacted = np.full((self.num_users, NUM_ITEMS), False)
        #user_row = np.arange(0, self.num_users)
        items = np.concatenate((self.recommend(k=num_recommended), np.random.choice(next(self.new_items_iter), size=(self.num_users, num_new_items))), axis=1)
        np.random.shuffle(items.T)
        if random_preference is True:
            preference = np.random.randint(0, num_items_per_iter, size=(self.num_users))
        interactions = items[np.arange(items.shape[0], dtype=int), preference]
        interactions = self._generate_interaction_matrix(interactions)
        self._store_interaction(interactions)
        self.measure_equilibrium(interactions)
        #if np.all(check):
        #    continue
        # TODO: From here on, some user(s) has already interacted with the assigned item

    def recommend(self, k=1):
        # TODO: what if I consistently only do k=1? In that case I might want to think of just sorting once
        #return self.s_t.argsort()[-k:][::-1]
        # Assume s_t two-dimensional
        return self.s_t.argsort()[:,::-1][:,0:k]

    def get_delta(self):
        return self.measurements.get_delta()


if __name__ == '__main__':
    rec = PopularityRecommender(const.NUM_USERS, const.NUM_ITEMS)
    # Startup
    rec.interact_startup(const.NUM_STARTUP_ITER, num_items_per_iter=const.NUM_ITEMS_PER_ITER, random_preference=True)
    rec.train()

    # Runtime
    for t in range(const.TIMESTEPS):
        rec.interact(int(const.NUM_ITEMS_PER_ITER / 2), int(const.NUM_ITEMS_PER_ITER / 2), True)
        rec.train()
    delta_t = rec.get_delta()
    plt.plot(np.arange(len(delta_t)), delta_t)
    plt.show()