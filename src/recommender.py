import numpy as np
from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt

# Recommender systems: abstract class
class Recommender(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, num_users, num_items, num_items_per_iter=10,
        randomize_recommended=True, num_recommended=None, num_new_items=None,
        user_preferences=False, measurements=None, debugger=None):
        # NOTE: Children classes must implement user_profiles (theta_t) and item_attributes (beta_t)
        self.scores = None
        self.measurements = measurements
        self.num_users = num_users
        self.num_items = num_items
        self.num_items_per_iter = num_items_per_iter
        self.debugger = debugger.get_logger(__name__.upper())
        # Matrix keeping track of the items consumed by each user
        self.indices = np.tile(np.arange(num_items), (num_users, 1))
        if not randomize_recommended:
            self.num_recommended = num_recommended
            self.num_new_items = num_new_items
        else:
            self.randomize_recommended = True
        # NOTE user_preferences either accepts False (randomize user preferences),
        # or it accepts a matrix of user preferences
        self.user_preferences = user_preferences
        self.debugger.log('Recommender system ready')
        self.user_vector = np.arange(num_users, dtype=int)
        self.item_vector = np.arange(2, num_items_per_iter + 2, dtype=int)
        self.debugger.log('Num items: %d' % self.num_items)
        self.debugger.log('Users: %d' % self.num_users)
        self.debugger.log('Items per iter: %d' % self.num_items_per_iter)
        self.debugger.log('Actual scores given by users (rows) to items (columns), unknown to system:\n' + str(self.user_preferences.get_preference_matrix()))

    # Train recommender system
    def train(self, user_profiles=None, item_attributes=None):
        user_profiles = user_profiles if user_profiles is not None else self.user_profiles
        item_attributes = item_attributes if item_attributes is not None else self.item_attributes
        self.scores = np.dot(user_profiles, item_attributes)
        self.debugger.log('System updates predicted scores given by users (rows) to items (columns):\n' + str(self.scores))

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
            self.debugger.log('System picked these items (cols) randomly for each user (rows):\n' + str(new_items))
        
        if recommended is not None and num_new_items:
            self.debugger.log('System recommended these items (cols) to each user (rows):\n' + str(recommended))
            items = np.concatenate((recommended, new_items), axis=1)
        elif num_new_items:
            items = new_items
        else:
            self.debugger.log('System recommended these items (cols) to each user (rows):\n' + str(recommended))
            items = recommended
        
        np.random.shuffle(items.T)
        #self.debugger.log("System recommends these items (columns) to each user (rows):\n" + str(items))
        if self.user_preferences is None:
            preference = np.random.randint(num_new_items, size=(self.num_users))
        else:
            preference = self.user_preferences.get_user_choices(items, self.user_vector)
            #print(preference)
        #print(preference.shape)
        interactions = items[self.user_vector, preference]
        self.debugger.log("Users choose the following items respectively:\n" + str(interactions))
        self.indices[self.user_vector, interactions] = -1
        return interactions

    @abstractmethod
    def run(self, plot=True, step=None, startup=False):
        #assert(np.count_nonzero(self.scores))
        self.interact(plot=plot, step=step, startup=startup)
