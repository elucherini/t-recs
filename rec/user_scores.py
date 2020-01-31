import numpy as np

from .debug import VerboseMode
from .stats import Distribution
from .utils import normalize_matrix

'''
'' Class representing the scores assigned to each item by the users.
'' These scores are unknown to the system.
'' Actual user scores are represented in the system by a |U|x|I| matrix,
'' where actual_scores(ui) is the actual score assigned by user u to item i.
'''
class ActualUserScores(VerboseMode):
    '''
    '' @num_users: number of users in the system
    '' @item_representation: description of items known by both users and system. The 
    ''      dimensions of this matrix must be |A|x|I|.
    '' @normalize: set to False if user_profiles should not be normalized
    '' @verbose: if True, enable verbose mode
    '' @distribution: Distribution instance for random sampling of user profiles
    '''
    def __init__(self, actual_user_profiles=None, actual_user_scores=None, num_users=None, item_representation=None, 
        normalize=True, verbose=False, distribution=None):
        # Initialize verbose mode
        super().__init__(__name__.upper(), verbose)

        # default if distribution not specified or invalid type
        if distribution is None or not isinstance(distribution, Distribution):
            self.distribution = Distribution('norm', non_negative=True)
            self.log('Initialize default normal distribution with no size')
        else:
            self.distribution = distribution

        self.normalize = normalize

        # Set up ActualUserScores instance
        if item_representation is None or num_users is None:
            self.user_profiles = None
            self.actual_scores = None
            self.log('Initialized empty ActualUserScores instance')
            return

        # Compute actual user profiles (|U|x|A|) and actual user scores (|U|x|I|)
        self.actual_scores = self.compute_actual_scores(item_representation=item_representation,
            num_users=num_users)
        self._print_verbose()


    '''
    '' Internal function to compute actual user scores
    '' @user_profiles: number of users
    '' @item_representation: description of items
    '''
    def _compute_actual_scores(self, user_profiles, item_representation):
        # Compute actual user scores
        scores = np.dot(user_profiles, item_representation)
        if self.normalize:
            scores = normalize_matrix(scores)
        return scores

    '''
    '' Compute actual user profiles unknown to system and actual user scores
    '' based on those profiles.
    '' @item_representation: description of items
    '' @num_users: int, number of users in the system (|U|)
    '' @distribution: Distribution instance for random sampling of user profiles
    '' @normalize: if True, enable verbose mode
    '''
    def compute_actual_scores(self, item_representation, num_users=100, distribution=None):
        # Error if item_representation is invalid
        if not isinstance(item_representation, np.ndarray) or item_representation.ndim != 2:
            raise TypeError("item_representation must be a |A|x|I| matrix and can't \
                            be None")
        # Use distribution if specified, otherwise default to self.distribution
        if distribution is not None and isinstance(distribution, Distribution):
            self.distribution = distribution
        # Compute user profiles (|U|x|A|)
        num_attr = item_representation.shape[0]
        self.user_profiles = self.distribution.compute(size=(num_users, num_attr))
        actual_scores = self._compute_actual_scores(self.user_profiles, 
                                                    item_representation)
        self.actual_scores = actual_scores
        return self.actual_scores

    '''
    '' Compute user profiles based on the new items in the system and
    '' update actual_scores.
    '' This function should be called when new items are introduced at runtime
    '' @item_representation: description of items
    '''
    def expand_items(self, item_representation):
        # Compute actual user scores for new items
        assert(item_representation.shape[0] == self.user_profiles.shape[1])
        if item_representation.shape[1] < self.actual_scores.shape[1]:
            raise ValueError("Wrong size for item_representation")
        new_scores = self._compute_actual_scores(self.user_profiles,
            item_representation)
        # Update actual user scores
        self.actual_scores = new_scores
        self._print_verbose()

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
    def get_user_feedback(self, items, user_vector):
        m = self.actual_scores[user_vector.reshape((items.shape[0], 1)), items]
        self.log('User scores for given items are:\n' + str(m))
        sorted_user_preferences = m.argsort()[:,::-1][:,0]
        interactions = items[user_vector, sorted_user_preferences]
        self.log("Users interact with the following items respectively:\n" + \
            str(interactions))
        return interactions

    '''
    '' Utility function for debug
    '''
    def _print_verbose(self):
        best_items = self.actual_scores.argmax(axis=1)
        self.log('Shape: ' + str(self.actual_scores.shape))

        self.log('Actual scores given by users (rows) to items (columns), ' + \
            'unknown to system:\n' + str(self.get_actual_user_scores()))
