import numpy as np
import seaborn

import matplotlib.pyplot as plt
from debug import Debug

'''
'' Class representing the scores assigned to each item by the users.
'' These scores are unknown to the system.
'' Actual user scores are represented in the system by a |U|x|I| matrix,
'' where s_ui is the actual score assigned user u to item i.
'''
class ActualUserScores():
    '''
    '' @num_users: number of users in the system
    '' @item_representation: description of items known by both users and system
    '' @debugger: Debug instance
    '' @distribution: distribution for random sampling
    '' @normalize: set to False if user_profiles should not be normalized
    '' @**kwargs: arguments of distribution (leave out size)
    '''
    def __init__(self, num_users, item_representation, debugger, distribution=np.random.normal, 
        normalize=True, **kwargs):
        kwargs['normalize'] = normalize
        self.actual_scores = self._compute_actual_scores(num_users, item_representation,
            distribution=distribution, **kwargs)
        self.debugger = debugger.get_logger(__name__.upper())
        self._print_debug()

    '''
    '' Internal function to compute user scores
    '' @num_users: number of users
    '' @item_representation: description of items
    '' @distribution: distribution for random sampling
    '' @**kwargs: arguments of distribution (leave out size)
    '''
    def _compute_actual_scores(self, num_users, item_representation, 
        distribution=np.random.normal, **kwargs):
        # Store value of normalize and remove from kwargs
        if 'normalize' in kwargs:
            normalize = kwargs.pop('normalize')
        else:
            normalize = True
        user_profiles = abs(distribution(**kwargs, 
            size=(num_users, item_representation.shape[0])))
        if normalize:
            user_profiles = user_profiles / user_profiles.sum(axis=1)[:,None]
        # Calculate actual user scores
        actual_scores = np.dot(user_profiles, item_representation)
        return actual_scores

    '''
    '' Compute user scores of new items
    '' This function should be called when new items are introduced at runtime
    '' @item_representation: description of items
    '' @num_new_items: number of items introduced at runtime
    '' @normalize: set to False if user_profiles should not be normalized
    '' @distribution: distribution for random sampling
    '' @**kwargs: arguments of distribution (leave out size)
    '''
    def expand_items(self, item_representation, num_new_items, normalize=True,
        distribution=np.random.normal, **kwargs):
        new_scores = self._compute_actual_scores(self.actual_scores.shape[0],
            item_representation[:,-num_new_items:], distribution=distribution,
            **kwargs)
        self.actual_scores = np.concatenate((self.actual_scores, new_scores),
            axis=1)
        self._print_debug()

    '''
    '' Return the actual user scores.
    '' If @user is not None, the function returns the actual scores for user u.
    '' @user: user id (index in the matrix)
    '' TODO: expand this
    '''
    def get_actual_user_scores(self, user=None):
        if user is None:
            return self.actual_scores
        else:
            return self.actual_scores[user, :]

    '''
    '' Return vector of user choices at a given timestep, s.t. element c_u(t) of
    '' the vector represents the index of the item selected by user u at time t.
    '' @items: recommended/new items provided by the system at the current timestep
    '' @user_vector: vector of user ids used for indexing
    '''
    def get_user_choices(self, items, user_vector):
        m = self.actual_scores[user_vector.reshape((items.shape[0], 1)), items]
        self.debugger.log('User scores for given items are:\n' + str(m))
        return m.argsort()[:,::-1][:,0]

    '''
    '' Utility function for debug
    '''
    def _print_debug(self):
        best_items = self.actual_scores.argmax(axis=1)
        self.debugger.log('Shape: ' + str(self.actual_scores.shape))
        self.debugger.pyplot_plot(best_items, np.arange(self.actual_scores.shape[1] + 1),
            plot_func=plt.hist, xlabel='Items', ylabel='# users who like item i the most',
            title='Histogram of users liking each item the most')
        plt.show()


# Unit test
if __name__ == '__main__':
    # Debugger module
    debugger = Debug(__name__.upper(), True)

    num_users = 3
    num_items = 5
    A = num_items
    num_new_items = 2

    # Random normalized representation
    #item_representation = np.random.randint(0, num_items, size=(num_items, A))
    #item_representation = item_representation / item_representation.sum(axis=1)[:,None]

    # Random binary item representation
    item_representation = np.random.binomial(1, .3, size=(num_items, A))

    # Print item representation
    logger = debugger.get_logger(__name__.upper())
    logger.log("Items:\n" + str(item_representation))
    actual_scores = ActualUserScores(num_users, item_representation.T, debugger,
        normalize=True, loc=0, scale=3)
    logger.log("Actual user score:\n" + str(actual_scores.actual_scores))
    new_items = np.random.binomial(1,.3, size=(num_new_items, A))
    logger.log("Adding %d new items:\n%s" % (num_new_items, str(new_items)))
    actual_scores.expand_items(np.concatenate((actual_scores.actual_scores, new_items), 
        axis=0), num_new_items)
    logger.log("Actual user score:\n" + str(actual_scores.actual_scores))
