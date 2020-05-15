from rec.models import BaseRecommender
import numpy as np
from rec.metrics import MSEMeasurement
from rec.components import BinarySocialGraph
from rec.random import SocialGraphGenerator
from rec.utils import get_first_valid, is_array_valid_or_none, is_equal_dim_or_none, all_none, is_valid_or_none

class SocialFiltering(BaseRecommender, BinarySocialGraph):
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
        verbose=False, num_items_per_iter=10, num_new_items=30, seed=None):
        # Give precedence to user_representation, otherwise build empty one

        if all_none(user_representation, num_users):
            raise ValueError("user_representation and num_users can't be all None")
        if all_none(item_representation, num_items):
            raise ValueError("item_representation and num_items can't be all None")

        if not is_array_valid_or_none(user_representation, ndim=2):
            raise ValueError("user_representation is invalid")
        if not is_array_valid_or_none(item_representation, ndim=2):
            raise ValueError("item_representation is not valid")
        num_items = get_first_valid(getattr(item_representation, 'shape',
                                            [None, None])[1],
                                    num_items)

        num_users = get_first_valid(getattr(user_representation,
                                            'shape', [None])[0],
                                    getattr(item_representation,
                                            'shape', [None])[0],
                                             num_users)

        if user_representation is None:
            import networkx as nx
            user_representation = SocialGraphGenerator.generate_random_graph(n=num_users,
                                                                        p=0.3, seed=seed,
                                                    graph_type=nx.fast_gnp_random_graph)
            #np.diag(np.diag(np.ones((num_users, num_users),
            #                                              dtype=int)))
        if item_representation is None:
            item_representation = np.zeros((num_users, num_items), dtype=int)

        if not is_equal_dim_or_none(getattr(user_representation, 'shape', [None])[0],
                                getattr(user_representation, 'shape', [None, None])[1],
                                num_users):
            raise ValueError("user_representation must be a square matrix")
        if not is_equal_dim_or_none(getattr(user_representation, 'shape',
                                            [None, None])[1],
                                getattr(item_representation, 'shape', [None])[0],
                                num_users):
            raise ValueError("user_representation.shape[1] should be the same as " + \
                             "item_representation.shape[0]")
        if not is_equal_dim_or_none(getattr(item_representation, 'shape',
                                            [None, None])[1],
                                    num_items):
            raise ValueError("item_representation.shape[1] should be the same as " + \
                             "num_items")

        measurements = [MSEMeasurement()]
        # Initialize recommender system
        BaseRecommender.__init__(self, user_representation, item_representation,
                             actual_user_scores, num_users, num_items,
                             num_items_per_iter, num_new_items, seed=seed,
                             measurements=measurements, verbose=verbose)


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
        interactions_per_user[self.actual_users._user_vector, interactions] = 1
        assert(interactions_per_user.shape == self.item_attributes.shape)
        self.item_attributes = np.add(self.item_attributes, interactions_per_user)
