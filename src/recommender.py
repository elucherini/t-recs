import numpy as np
from abc import ABCMeta, abstractmethod

# Recommender systems: abstract class
class Recommender(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, num_users, num_items, num_startup_iter=10, num_items_per_iter=10,
        randomize_recommended=True, num_recommended=None, num_new_items=None,
        user_preference=False, measurements=None):
        # NOTE: Children classes: implement theta, beta
        self.s_t = None
        self.measurements = measurements
        self.new_items_iter = None
        self.num_users = num_users
        self.num_items = num_items
        self.num_startup_iter = num_startup_iter
        self.num_items_per_iter = num_items_per_iter
        if not randomize_recommended:
            self.num_recommended = num_recommended
            self.num_new_items = num_new_items
        else:
            self.randomize_recommended = True
        # NOTE user_preference either accepts False (randomize user preferences),
        # or it accepts an array of user preferences
        self.user_preference = user_preference

    # Return matrix that can be stored to train a model
    def _generate_interaction_matrix(self, interactions):
        tot_interactions = np.zeros(self.num_items)
        np.add.at(tot_interactions, interactions, 1)
        return tot_interactions

    # Train recommender system
    def train(self):
        self.s_t = np.dot(self.theta_t, self.beta_t)

    # TODO: what if I consistently only do k=1? In that case I might want to think of just sorting once
    #return self.s_t.argsort()[-k:][::-1]
    # Assume s_t two-dimensional
    def recommend(self, k=1):
        return self.s_t.argsort()[:,::-1][:,0:k]

    @abstractmethod
    def interact(self, user_vector, recommended, num_new_items):
        # Current assumptions:
        # 1. Interleave new items and recommended items
        # 2. Fixed number of new/recommended items
        # 3. New items are chosen randomly from the same set of num_items_per_iter * const.CONSTANT items
        # 4. Each user interacts with different elements depending on preference
        # 5. Train after every interaction
        # TODO: each user can interact with each element at most once
        #interacted = np.full((self.num_users, NUM_ITEMS), False)
        #user_row = np.arange(0, self.num_users)
        #items = np.concatenate((recommended, np.random.choice(next(self.new_items_iter), size=(self.num_users, num_new_items))), axis=1)
        items = np.concatenate((recommended, np.random.choice(self.num_items, size=(self.num_users, num_new_items))), axis=1)
        assert(items.shape[1] == self.num_items_per_iter)
        np.random.shuffle(items.T)
        if self.user_preference is False:
            preference = np.random.randint(0, self.num_items_per_iter, size=(self.num_users))
        interactions = items[user_vector, preference]
        return self._generate_interaction_matrix(interactions)
        #self._store_interaction(interactions)
        #self.measure_equilibrium(interactions)
        #if np.all(check):
        #    continue
        # TODO: From here on, some user(s) has already interacted with the assigned item

    @abstractmethod
    def interact_startup(self, constant):
        # Current assumptions:
        # 1. First (num_startup_iter * num_items_per_iter) items presented for startup
        # 2. New  num_items_per_iter items at each interaction, no recommendations
        new_items = np.arange(self.num_startup_iter * self.num_items_per_iter, self.num_items).reshape(-1, self.num_items_per_iter * constant)
        self.new_items_iter = iter(new_items)
        if self.user_preference is False:
            preference = np.zeros(self.num_users * self.num_startup_iter, dtype=int)
        index = 0
        for t in range(1, self.num_startup_iter + 1):
            if self.user_preference is False:
                preference[index:index+self.num_users] = np.random.randint((t-1) * self.num_items_per_iter, t * self.num_items_per_iter, size=(self.num_users))
            index += self.num_users
        return self._generate_interaction_matrix(preference)