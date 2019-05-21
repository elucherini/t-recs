import numpy as np
from abc import ABCMeta, abstractmethod

# Recommender systems: abstract class
class Recommender(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        self.beta_t = None
        self.theta_t = None
        self.s_t = None


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
    def interact(self, user_vector, num_recommended, num_new_items, random_preference, 
        preference, recommended):
        # Current assumptions:
        # 1. Interleave new items and recommended items
        # 2. Fixed number of new/recommended items
        # 3. New items are chosen randomly from the same set of num_items_per_iter * const.CONSTANT items
        # 4. Each user interacts with different elements depending on preference
        # 5. Train after every interaction
        # TODO: each user can interact with each element at most once
        # TODO: consider user preferences that are not random
        num_items_per_iter = num_new_items + num_recommended
        #interacted = np.full((self.num_users, NUM_ITEMS), False)
        #user_row = np.arange(0, self.num_users)
        items = np.concatenate((recommended, np.random.choice(next(self.new_items_iter), size=(self.num_users, num_new_items))), axis=1)
        np.random.shuffle(items.T)
        if random_preference is True:
            preference = np.random.randint(0, num_items_per_iter, size=(self.num_users))
        interactions = items[user_vector, preference]
        return self._generate_interaction_matrix(interactions)
        #self._store_interaction(interactions)
        #self.measure_equilibrium(interactions)
        #if np.all(check):
        #    continue
        # TODO: From here on, some user(s) has already interacted with the assigned item

    @abstractmethod
    def interact_startup(self, num_startup_iter, num_items_per_iter, random_preference, preference, constant):
        # Current assumptions:
        # 1. First (num_startup_iter * num_items_per_iter) items presented for startup
        # 2. New  num_items_per_iter items at each interaction, no recommendations
        # 3. TODO: consider user preferences that are not random
        new_items = np.arange(num_startup_iter * num_items_per_iter, self.num_items).reshape(-1, num_items_per_iter * constant)
        self.new_items_iter = iter(new_items)
        if random_preference is True:
            preference = np.zeros(self.num_users * num_startup_iter, dtype=int)
        index = 0
        for t in range(1, num_startup_iter + 1):
            if random_preference is True:
                preference[index:index+self.num_users] = np.random.randint((t-1) * num_items_per_iter, t * num_items_per_iter, size=(self.num_users))
            index += self.num_users
        return self._generate_interaction_matrix(preference)