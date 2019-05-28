import numpy as np
import random
from abc import ABCMeta, abstractmethod
from recommender import Recommender
from measurements import Measurements
import matplotlib.pyplot as plt

class PopularityRecommender(Recommender):
    def __init__(self, num_users, num_items, num_startup_iter=10, num_items_per_iter=10,
        randomize_recommended=True, num_recommended=None, num_new_items=None,
        user_preference=False):
        # TODO: check on invalid parameters
        self.user_profiles = np.ones((num_users, 1), dtype=int)
        self.item_attributes = np.zeros((1, num_items), dtype=int)
        super().__init__(num_users, num_items, num_startup_iter, num_items_per_iter,
        randomize_recommended, num_recommended, num_new_items,
        user_preference, Measurements(num_items))

    def _store_interaction(self, interactions):
        self.item_attributes = np.add(self.item_attributes, interactions)

    def train(self):
        return super().train()

    def measure_equilibrium(self, interactions, plot=False):
        return self.measurements.measure_equilibrium(interactions, plot)

    def interact(self, plot=False, startup=False):
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
        interactions = super().interact(recommended, num_new_items)
        self._store_interaction(interactions)
        self.measure_equilibrium(interactions, plot=plot)

    # Recommends items proportional to their popularity
    def recommend(self, k=1):
        indices_prime = self.indices[np.where(self.indices>=0)].reshape((self.num_users, -1))
        if k > indices_prime.shape[1]:
            # TODO exception
            print('recommend Nope')
            return
        row = np.repeat(self.user_vector, indices_prime.shape[1]).reshape((self.num_users, -1))
        s_filtered = self.scores[row, indices_prime]
        permutation = s_filtered.argsort()
        rec = indices_prime[row, permutation]
        probabilities = np.arange(1, rec.shape[1] + 1)
        probabilities = probabilities/probabilities.sum()
        picks = np.random.choice(permutation[0], p=probabilities, size=(self.num_users, k))
        return rec[np.repeat(self.user_vector, k).reshape((self.num_users, -1)), picks]

    def get_heterogeneity(self):
        return self.measurements.get_delta()

    def startup_and_train(self, timesteps=50, debug=False):
        assert(np.count_nonzero(self.scores) == 0)
        self.measurements.set_delta(timesteps)
        for t in range(timesteps):
            plot = False
            if debug:
                if t == 0: #or t == timesteps - 1:
                    plot=True
            self.interact(plot=plot, startup=True)
        self.train()
        #plt.plot(np.arange(self.scores.shape[1]), sorted(self.scores[0]))
        #plt.show()

    def run(self, timesteps=50, train=True, debug=False):
        self.measurements.set_delta(timesteps)
        for t in range(timesteps):
            plot = False
            if debug:
                if t == 0:#% 50 == 0 or t == timesteps - 1:
                    plot=True
            self.interact(plot=plot)
            plt.show()
            if train:
                self.train()