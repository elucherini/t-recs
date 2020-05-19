import numpy as np

from rec.utils import VerboseMode, normalize_matrix
from rec.random import Generator
from .base_component import BaseComponent

class Users(BaseComponent):
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
    """
    def __init__(self, actual_user_profiles=None, actual_user_scores=None,
                 interact_with_items=None, size=None, num_users=None,
                 verbose=False, seed=None):
        # general input checks
        if actual_user_profiles is not None:
            if not isinstance(actual_user_profiles, (list, np.ndarray)):
                raise TypeError("actual_user_profiles must be a list or numpy.ndarray")
        if interact_with_items is not None and not callable(interact_with_items):
            raise TypeError("interact_with_items must be callable")
        if actual_user_profiles is None and size is None:
            raise ValueError("actual_user_profiles and size can't both be None")
        if actual_user_profiles is None and not isinstance(size, tuple):
            raise TypeError("size must be a tuple, is %s" % type(size))
        if actual_user_scores is not None:
            if not isinstance(actual_user_scores, (list, np.ndarray)):
                raise TypeError("actual_user_profiles must be a list or numpy.ndarray")
        if actual_user_profiles is None and size is not None:
            actual_user_profiles = Generator(seed=seed).normal(size=size)
        self.actual_user_profiles = np.asarray(actual_user_profiles)
        self.interact_with_items = interact_with_items
        # this will be initialized by the system
        self.actual_user_scores = None
        if num_users is not None:
            self._user_vector = np.arange(num_users, dtype=int)
        self.name = 'Actual Users'
        BaseComponent.__init__(self, verbose=verbose, init_value=self.actual_user_scores)


    def compute_user_scores(self, train_function, *args, **kwargs):
        # TODO: this must be called by expand_items
        if not callable(train_function):
            raise TypeError("train_function must be callable")
        self.actual_user_scores = train_function(user_profiles=self.actual_user_profiles,
                                                 *args, **kwargs)
        self.store_state()

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
            return self.actual_user_scores
        else:
            return self.actual_user_scores[user, :]

    def get_user_feedback(self, *args, **kwargs):
        """Generates user interactions at a given timestep.

            Args:
                items (:obj:`numpy.ndarray`): A |U|x|num_items_per_iter| matrix with recommendations and
                    new items.

            Returns:
                Array of interactions s.t. element interactions_u(t) represents the
                index of the item selected by user u at time t. Shape: |U|
        """
        if self.interact_with_items is not None:
            return self.interact_with_items(*args, **kwargs)
        items = kwargs.pop('items', None)
        if items is None:
            raise ValueError("Items can't be None")
        reshaped_user_vector = self._user_vector.reshape((items.shape[0], 1))
        user_interactions = self.actual_user_scores[reshaped_user_vector, items]
        self.log('User scores for given items are:\n' + str(user_interactions))
        sorted_user_preferences = user_interactions.argsort()[:,::-1][:,0]
        interactions = items[self._user_vector, sorted_user_preferences]
        self.log("Users interact with the following items respectively:\n" + \
            str(interactions))
        return interactions

    def store_state(self):
        self.component_data.append(np.copy(self.actual_user_scores))


        #def compute_actual_scores(self, item_representation, num_users, distribution=None):
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
        #if item_representation.ndim != 2:
        #    item_representation = np.tile(item_representation, (num_users, 1))


