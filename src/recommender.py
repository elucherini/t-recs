import numpy as np
from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt

# Recommender systems: abstract class
class Recommender(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, num_users, num_items, num_items_per_iter=10,
        randomize_recommended=True, num_recommended=None, num_new_items=None,
        actual_user_scores=False, measurements=None, debugger=None):
        # NOTE: Children classes must implement user_profiles and item_attributes
        self.debugger = debugger.get_logger(__name__.upper())
        # set self.predicted_scores
        self.train(normalize=False)
        self.measurements = measurements
        self.num_users = num_users
        self.num_items = num_items
        self.num_items_per_iter = num_items_per_iter
        # Matrix keeping track of the items consumed by each user
        self.indices = np.tile(np.arange(num_items), (num_users, 1))
        if not randomize_recommended:
            self.num_recommended = num_recommended
            self.num_new_items = num_new_items
        else:
            self.randomize_recommended = True
        # NOTE user_preferences either accepts False (randomize user preferences),
        # or it accepts a matrix of user preferences
        self.actual_user_scores = actual_user_scores
        self.debugger.log('Recommender system ready')
        self.user_vector = np.arange(num_users, dtype=int)
        self.item_vector = np.arange(2, num_items_per_iter + 2, dtype=int)
        self.debugger.log('Num items: %d' % self.num_items)
        self.debugger.log('Users: %d' % self.num_users)
        self.debugger.log('Items per iter: %d' % self.num_items_per_iter)

    # Train recommender system
    def train(self, user_profiles=None, item_attributes=None):
        if user_profiles is not None:
            user_profiles = user_profiles 
        else:
            user_profiles = self.user_profiles

        if item_attributes is not None:
            item_attributes = item_attributes
        else:
            item_attributes = self.item_attributes
        self.predicted_scores = np.dot(user_profiles, item_attributes)
        self.debugger.log('System updates predicted scores given by users (rows) ' + \
            'to items (columns):\n' + str(self.predicted_scores))

    @abstractmethod
    def _expand_items(self, num_new_items):
        new_indices = np.tile(np.arange(self.num_items, self.num_items + num_new_items), 
            (self.num_users,1))
        self.indices = np.concatenate((self.indices, new_indices), axis=1)
        self.num_items += num_new_items
        self.debugger.log('Successfully added %d new items' % num_new_items)

    # TODO: what if I consistently only do k=1? In that case I might want to think of just sorting once
    #return self.scores.argsort()[-k:][::-1]
    # Assume scores two-dimensional
    @abstractmethod
    def recommend(self, k=1, indices_prime=None):
        if indices_prime is None:
            indices_prime = self.indices[np.where(self.indices>=0)]
            indices_prime = indices_prime.reshape((self.num_users, -1))
        #self.debugger.log('Indices_prime:\n' + str(indices_prime))
        if indices_prime.size == 0 or k > indices_prime.shape[1]:
            self.debugger.log('Insufficient number of items left!')
            self._expand_items()
            indices_prime = self.indices[np.where(self.indices>=0)]
            indices_prime = indices_prime.reshape((self.num_users, -1))
        row = np.repeat(self.user_vector, indices_prime.shape[1])
        row = row.reshape((self.num_users, -1))
        #self.debugger.log('row:\n' + str(row))
        self.debugger.log('Row:\n' + str(row))
        self.debugger.log('Indices_prime:\n' + str(indices_prime))
        s_filtered = self.predicted_scores[row, indices_prime]
        #self.debugger.log('s_filtered\n' + str(s_filtered))
        permutation = s_filtered.argsort()
        #self.debugger.log('permutation\n' + str(permutation))
        rec = indices_prime[row, permutation]
        probabilities = np.logspace(0.0, rec.shape[1]/10.0, num=rec.shape[1], base=2)
        probabilities = probabilities/probabilities.sum()
        self.debugger.log('Items ordered by preference for each user:\n' + str(rec))
        picks = np.random.choice(permutation.shape[1], p=probabilities, size=(self.num_users, k)) 
        #self.debugger.log('recommendations\n' + str(rec[np.repeat(self.user_vector, k).reshape((self.num_users, -1)), picks]))
        #print(self.predicted_scores.argsort()[:,::-1][:,0:5])
        return rec[np.repeat(self.user_vector, k).reshape((self.num_users, -1)), picks]
        #return self.predicted_scores.argsort()[:,::-1][:,0:k]

    @abstractmethod
    def interact(self, num_recommended, num_new_items):
        if num_recommended == 0 and num_new_items == 0:
            # TODO throw exception here
            print("Nope")
            return
        if num_recommended > 0:
            recommended = self.recommend(k=num_recommended)
            assert(num_recommended == recommended.shape[1])
            assert(recommended.shape[0] == self.num_users)
            self.debugger.log('System recommended these items (cols) to each user ' +\
                '(rows):\n' + str(recommended))
        else:
            recommended = None
        indices_prime = self.indices[np.where(self.indices>=0)]
        indices_prime = indices_prime.reshape((self.num_users, -1))
        # Current assumptions:
        # 1. Interleave new items and recommended items
        # 2. Each user interacts with one element depending on preference
        assert(np.count_nonzero(self.indices == -1) % self.num_users == 0)
        if indices_prime.shape[1] < num_new_items:
            self.debugger.log('Insufficient number of items left!')
            self._expand_items()
            indices_prime = self.indices[np.where(self.indices>=0)]
            indices_prime = indices_prime.reshape((self.num_users, -1))

        if num_new_items:
            col = np.random.randint(indices_prime.shape[1], size=(self.num_users, num_new_items))
            row = np.repeat(self.user_vector, num_new_items).reshape((self.num_users, -1))
            new_items = indices_prime[row, col]
            self.debugger.log('System picked these items (cols) randomly for each user ' + \
                '(rows):\n' + str(new_items))
        
        if num_recommended and num_new_items:
            items = np.concatenate((recommended, new_items), axis=1)
        elif num_new_items:
            items = new_items
        else:
            items = recommended
        np.random.shuffle(items.T)
        #self.debugger.log("System recommends these items (columns) to each user (rows):\n" + str(items))
        if self.actual_user_scores is None:
            preference = np.random.randint(num_new_items, size=(self.num_users))
        else:
            preference = self.actual_user_scores.get_user_choices(items, self.user_vector)
            #print(preference)
        #print(preference.shape)
        interactions = items[self.user_vector, preference]
        self.debugger.log("Users choose the following items respectively:\n" + \
            str(interactions))
        self.indices[self.user_vector, interactions] = -1
        return interactions

    @abstractmethod
    def run(self, plot=True, step=None, startup=False):
        #assert(np.count_nonzero(self.predicted_scores))
        self.interact(plot=plot, step=step, startup=startup)
