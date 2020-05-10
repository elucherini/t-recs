from .recommender import Recommender
import numpy as np
from .measurement import DiffusionTreeMeasurement
from .socialgraph import BinarySocialGraph

class SocialFiltering(Recommender, BinarySocialGraph):
    """A customizable social-filtering recommendation system.

        With social filtering, users are presented items that were previously liked by other users in their
        social networks.

        The social network is represented by a |U|x|U| matrix, where |U| is the number of users in the system.
        For each pair of users u and v, entry (u,v) defines whether u "follows"/is connected with v. This
        can be a binary relationship or a score that measures how likely u is to engage with content that
        v has previously interacted with.

        Please note that the follow/unfollow and add_friends/remove_friends methods assume a binary social graph.

        Item attributes are represented by a |U|x|I| matrix, where |I| is the number of items in the system.
        For each item i and user u, we define a score that determines the interactions u had with i. Again, this
        could just be a binary relationship.

        Args:
            num_users (int, optional): The number of users |U| in the system.
            num_items (int, optiona;): The number of items |I| in the system.
            item_representation (:obj:`numpy.ndarray`, optional): A |U|x|I| matrix representing the similarity
                between each item and attribute. If this is not None, num_items is
                ignored.
            user_representation (:obj:`numpy.ndarray`, optional): A |U|x|U| matrix representing each user's
                social network: if user_representation[u,v] > 0, then user u "follows" user v. If this
                is not None, num_users is ignored.
            actual_user_scores (:obj:`numpy.ndarray`, optional): A |U|x|I| matrix representing the real
                user scores. This matrix is *not* used for recommendations. This is
                only kept for measurements and the system is unaware of it.
            verbose (bool, optional): If True, enables verbose mode. Disabled by default.
            num_items_per_iter (int, optional): Number of items presented to the user per iteration.
            num_new_items (int, optional): Number of new items that the systems add if it runs out
                of items that the user can interact with.

        Attributes:
            Inherited by :class:`Recommender`

        Examples:
            SocialFiltering can be instantiated with no arguments -- in which case, it will
            be initialized with the default parameters and the item/user representation will be
            initialized to zero. This means that a user starts with no followers/users they follow,
            and that there have been no previous interactions for this set of users.

            >>> sf = SocialFiltering()
            >>> sf.user_profiles.shape
            (100, 100)   # <-- 100 users (default)
            >>> sf.item_attributes.shape
            (99, 1250) # <-- 100 users (default), 1250 items (default)

            This class can be customized either by defining the number of users/items
            in the system:

            >>> sf = SocialFiltering(num_users=1200, num_items=5000)
            >>> sf.item_attributes.shape
            (1200, 5000) # <-- 1200 users, 5000 items

            >>> sf = ContentFiltering(num_users=50)
            >>> sf.item_attributes.shape
            (50, 1250) # <-- 50 users, 1250 items (default)

            Or by generating representations for items and/or users:
            # Items are uniformly distributed. We "indirectly" define 100 users.
            >>> item_representation = np.random.randint(0, 1, size=(100, 200))
            # Social networks are drawn from a binomial distribution. This representation also uses 100 users.
            # Note: For this to be meaningful, make sure that the diagonal is all set to 1.
            >>> sf = istribution(distr_type='powerlaw').compute(a=1.16, size=(100, 100)).compute())
            >>> sf = SocialFiltering(item_representation=item_representation, user_representation=user_representation)
            >>> sf.item_attributes.shape
            (100, 200)
            >>> sf.user_profiles.shape
            (100, 100)

            Note that user and item representations have the precedence over the number of
            users/items specified at initialization. For example:
            >>> sf = SocialFiltering(num_users=50, user_representation=user_representation)
            >>> sf.item_attributes.shape
            (100, 200) # <-- 100 users, 200 items. num_users was ignored because user_representation was specified.

            The same happens with the number of items or users and item representations.

            >>> sf = SocialFiltering(num_users=1400, item_representation=item_representation)
            >>> sf.item_attributes.shape
            (100, 200) # <-- 100 attributes, 200 items. num_users was ignored.
            >>> cf.user_profile.shape
            (100, 100) # <-- 100 users (as implicitly specified by item_representation)

        """
    def __init__(self, num_users=100, num_items=1250,
        item_representation=None, user_representation=None, actual_user_scores=None,
        verbose=False, num_items_per_iter=10, num_new_items=30):
        # Give precedence to user_representation, otherwise build empty one
        if user_representation is None:
            if item_representation is not None:
                num_users = item_representation.shape[0]
            elif num_users is None:
                raise ValueError("num_users and user_representation can't be both None")
            user_representation = np.diag(np.diag(np.ones((num_users, num_users), dtype=int)))
        elif (item_representation is not None
            and item_representation.shape[0] != user_representation.shape[0]):
            raise ValueError("It should be user_representation.shape[0] (or shape[1])" + \
                                " == item_representation.shape[0]")
        elif (user_representation.shape[0] != user_representation.shape[1]):
            raise ValueError("It should be user_representation.shape[0]" + \
                                " == user_representation.shape[1]")
        else:
            num_users = user_representation.shape[0]
        assert(num_users is not None)
        assert(user_representation is not None)
        if user_representation is not None:
            assert(num_users == user_representation.shape[0] == user_representation.shape[1])
        # Give precedence to item_representation, otherwise build random one
        if item_representation is not None:
            num_items = item_representation.shape[1]
        else:
            if num_items is None:
                raise ValueError("num_items and item_representation can't be both None")
            else:
                item_representation = np.zeros((num_users, num_items), dtype=int)

        assert(num_items is not None)
        assert(item_representation is not None)
        # Initialize recommender system
        Recommender.__init__(self, user_representation, item_representation, actual_user_scores, num_users, num_items, num_items_per_iter, num_new_items, verbose=verbose)


    def _update_user_profiles(self, interactions):
        """ Private function that updates user profiles with data from
            latest interactions.

            Specifically, this function converts interactions into item attributes.
            For example, if user u has interacted with item i, then the i's attributes
            will be updated to increase the similarity between u and i.

        Args:
            interactions (numpy.ndarray): An array of item indices that users have
                interacted with in the latest step. Namely, interactions_u represents
                the index of the item that the user has interacted with.

        """
        interactions_per_user = np.zeros((self.num_users, self.num_items))
        interactions_per_user[self.user_vector, interactions] = 1
        assert(interactions_per_user.shape == self.item_attributes.shape)
        self.item_attributes = np.add(self.item_attributes, interactions_per_user)

    def train(self, user_profiles=None, item_attributes=None, normalize=True):
        """ Calls train method of parent class :class:`Recommender`.

            Args:
                normalize (bool, optional): set to True if the scores should be normalized,
            False otherwise.
        """
        if user_profiles is None:
            user_profiles = self.user_profiles
        if item_attributes is None:
            item_attributes = self.item_attributes
        assert(user_profiles.shape[1] == item_attributes.shape[0])
        return Recommender.train(self, user_profiles, item_attributes, normalize=normalize)
