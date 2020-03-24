import numpy as np

from .debug import VerboseMode
from .stats import Distribution
from .utils import normalize_matrix


class ActualUserScores(VerboseMode):
    """Class representing the scores assigned to each item by the users.
        These scores are unknown to the system.

        Actual user scores are represented in the system by a |U|x|I| matrix,
        where actual_scores(ui) is the actual score assigned by user u to item i.
    
    Args:
        num_users (int or None, optional): The number of users in the system.
        item_representation (numpy.ndarray or None, optional): description of items 
            known by both users and system. The dimensions of this matrix must be |A|x|I|.
        normalize (bool, optional): set to True if the scores should be normalized,
            False otherwise.
        verbose (bool, optional): If True, enables verbose mode. Disabled by default.
        distribution (:class:`Distribution` or None, optional): :class:`Distribution`
            instance for random sampling of user profiles.

    Attributes:
        Attributes inherited by :class:`VerboseMode`, plus:
        distribution (:class:`Distribution`): :class:`Distribution` instance for random
            sampling of user profiles.
        user_profiles (:obj:`numpy.ndarray): A |U|x|A| matrix representing the *real* 
            similarity between each item and attribute.
        actual_scores (:obj:`numpy.ndarray): A |U|x|I| matrix representing the *real*
            scores assigned by each user to each item.
        normalize (bool): If True, enables verbose mode.

    Examples:
        ActualUserScores can be instantiated with no arguments. However, it can't be
        initialized until item_representation is specified.

        >>> scores = ActualUserScores()
        >>> scores.actual_scores
        None

        Two examples of correct instantiations are the following:

        >>> item_representation = ...
        >>> scores = ActualUserScores()
        >>> scores.compute_actual_scores(item_representation)

        Or:
        >>> item_representation = ...
        >>> scores = ActualUserScores(item_representation=item_representation)

    Todo:
        * Allow numpy.ndarray in distribution argument
    """
    def __init__(self, num_users=None, item_representation=None, 
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
        self.user_profiles = None
        self.actual_scores = self.compute_actual_scores(item_representation=item_representation,
            num_users=num_users)
        self._print_verbose()

    def _compute_actual_scores(self, user_profiles, item_representation):
        """Internal function to compute user scores.

            Args:
                user_profiles (numpy.ndarray):
                item_representation(numpy.ndarray): A |A|x|I| matrix

            Returns:
                A |U|x|I| :obj:`numpy.ndarray` matrix representing the real scores.
        """
        # Compute actual user scores
        scores = np.dot(user_profiles, item_representation)
        if self.normalize:
            scores = normalize_matrix(scores)
        return scores

    def compute_actual_scores(self, item_representation, num_users, distribution=None):
        """Computes actual user profiles unknown to system and actual user scores based
            on those profiles.
            
            Args:
                item_representation (:obj:`numpy.ndarray`): A |A|x|I| matrix representing
                    the similarity between each item and attribute.
                num_users (int, optional): The number of users in the system.
                distribution (:class:`Distribution`, optional): Distribution instance for
                    random sampling of user profiles. If None, the function uses the
                    distribution attribute.

            Returns:
                A |U|x|I| matrix of scores representing the real user preferences on each
                item.

            Raises:
                TypeError: If item_representation is not a 2-dimensional :obj:`numpy.ndarray`.
        """
        # Error if item_representation is invalid
        if not isinstance(item_representation, np.ndarray) or item_representation.ndim != 2:
            raise TypeError("item_representation must be a |A|x|I| matrix and can't \
                            be None")
        # Use distribution if specified, otherwise default to self.distribution
        if distribution is not None and isinstance(distribution, Distribution):
            self.distribution = distribution
            self.user_profiles = self.distribution.compute(size=(num_users, num_attr))
        # Compute user profiles (|U|x|A|)
        num_attr = item_representation.shape[0]
        if self.user_profiles is None:
            self.user_profiles = self.distribution.compute(size=(num_users, num_attr))
        actual_scores = self._compute_actual_scores(self.user_profiles, 
                                                    item_representation)
        self.actual_scores = actual_scores
        return self.actual_scores

    def expand_items(self, item_representation):
        """Computes real user profiles based on the new items in the system and updates
            the actual user scores.
            This function is called by :class:`Recommender` when new items are introduced
            at runtime.

            Args:
                item_representation (:obj:`numpy.ndarray`): A |A|x|num_new_items| matrix
                    representing the similarity between each *new* item and attribute.

            Raises:
                ValueError: If the new item_representation contains fewer items than the
                current representation -- that is, if |I_old| > |I_new| 

        """
        # Compute actual user scores for new items
        assert(item_representation.shape[0] == self.user_profiles.shape[1])
        if item_representation.shape[1] < self.actual_scores.shape[1]:
            raise ValueError("Wrong size for item_representation")
        new_scores = self._compute_actual_scores(self.user_profiles,
            item_representation)
        # Update actual user scores
        self.actual_scores = new_scores
        self._print_verbose()

    def get_actual_user_scores(self, user=None):
        """Returns the actual user scores matrix.

            Args:
                user (int or numpy.ndarray or list): if not None, it specifies the user
                    id(s) for which to return the actual user scores.

            Returns:
                An array of user scores for each item.

            Todo:
                * Expand

        """
        if user is None:
            return self.actual_scores
        else:
            return self.actual_scores[user, :]

    def get_user_feedback(self, items, user_vector):
        """Generates user interactions at a given timestep.

            Args:
                items (:obj:`numpy.ndarray`): A |U|x|num_items_per_iter| matrix with recommendations and
                    new items.
                user_vector (:obj:`numpy.ndarray`): An array of length |U| s.t. user_vector_u = u
                    for u in U.

            Returns:
                Array of interactions s.t. element interactions_u(t) represents the
                index of the item selected by user u at time t. Size: |U|
        """
        m = self.actual_scores[user_vector.reshape((items.shape[0], 1)), items]
        self.log('User scores for given items are:\n' + str(m))
        sorted_user_preferences = m.argsort()[:,::-1][:,0]
        interactions = items[user_vector, sorted_user_preferences]
        self.log("Users interact with the following items respectively:\n" + \
            str(interactions))
        return interactions

    def _print_verbose(self):
        """ Utility function used for debugging. Prints information to log.
        """
        best_items = self.actual_scores.argmax(axis=1)
        self.log('Shape: ' + str(self.actual_scores.shape))

        self.log('Actual scores given by users (rows) to items (columns), ' + \
            'unknown to system:\n' + str(self.get_actual_user_scores()))
