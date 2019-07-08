import numpy as np
from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt

# Recommender systems: abstract class
class Recommender(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, num_users, num_items, num_startup_iter=10, num_items_per_iter=10,
        randomize_recommended=True, num_recommended=None, num_new_items=None,
        user_preference=False, measurements=None):
        # NOTE: Children classes must implement user_profiles (theta_t) and item_attributes (beta_t)
        self.scores = None
        self.measurements = measurements
        self.num_users = num_users
        self.num_items = num_items
        self.num_startup_iter = num_startup_iter
        self.num_items_per_iter = num_items_per_iter
        # Matrix keeping track of the items consumed by each user
        self.indices = np.tile(np.arange(num_items), (num_users, 1))
        if not randomize_recommended:
            self.num_recommended = num_recommended
            self.num_new_items = num_new_items
        else:
            self.randomize_recommended = True
        # NOTE user_preference either accepts False (randomize user preferences),
        # or it accepts an array of user preferences
        self.user_preference = user_preference
        self.user_vector = np.arange(num_users, dtype=int)

    # Train recommender system
    def train(self, user_profiles=None, item_attributes=None):
        user_profiles = user_profiles if user_profiles is not None else self.user_profiles
        item_attributes = item_attributes if item_attributes is not None else self.item_attributes
        self.scores = np.dot(user_profiles, item_attributes)

    # TODO: what if I consistently only do k=1? In that case I might want to think of just sorting once
    #return self.scores.argsort()[-k:][::-1]
    # Assume scores two-dimensional
    @abstractmethod
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
        #print(self.scores.argsort()[:,::-1][:,0:5])
        return rec[np.repeat(self.user_vector, k).reshape((self.num_users, -1)), picks]
        #return self.scores.argsort()[:,::-1][:,0:k]

    @abstractmethod
    def interact(self, recommended, num_new_items):
        # Current assumptions:
        # 1. Interleave new items and recommended items
        # 2. Each user interacts with one element depending on preference
        if recommended is None and num_new_items == 0:
            # TODO throw exception here
            print("Nope")
            return
        assert(np.count_nonzero(self.indices == -1) % self.num_users == 0)
        indices_prime = self.indices[np.where(self.indices>=0)].reshape((self.num_users, -1))
        if indices_prime.shape[1] < num_new_items:
            print("Not enough items")
            # TODO exception
            return
        if num_new_items:
            col = np.random.randint(indices_prime.shape[1], size=(self.num_users, num_new_items))
            row = np.repeat(self.user_vector, num_new_items).reshape((self.num_users, -1))
            new_items = indices_prime[row, col]
        
        if recommended is not None and num_new_items:
            items = np.concatenate((recommended, new_items), axis=1)
        elif num_new_items:
            items = new_items
        else:
            items = recommended
        
        np.random.shuffle(items.T)
        if self.user_preference is False:
            preference = np.random.randint(num_new_items, size=(self.num_users))
        interactions = items[self.user_vector, preference]
        self.indices[self.user_vector, interactions] = -1
        return interactions

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