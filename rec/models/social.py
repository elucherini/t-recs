"""
Social filtering recommender system, where users get shown items that were
interacted with by users in their social networks
"""
import networkx as nx
import numpy as np
from rec.metrics import MSEMeasurement
from rec.components import BinarySocialGraph
from rec.random import SocialGraphGenerator
from rec.utils import (
    get_first_valid,
    is_array_valid_or_none,
    all_besides_none_equal,
    all_none,
)
from .recommender import BaseRecommender


class SocialFiltering(BaseRecommender, BinarySocialGraph):
    """
    A customizable social filtering recommendation system.

    With social filtering, users are presented items that were previously
    liked by other users in their social networks.

    The social network is represented by a `|U|x|U|` matrix, where `|U|` is the
    number of users in the system. For each pair of users `u` and `v`, entry
    `[u,v]` defines whether `u` "follows"/is connected to `v`. This can be a
    binary relationship or a score that measures how likely `u` is to engage
    with content that `v` has previously interacted with.

    Please note that, in this class, the follow/unfollow and
    add_friends/remove_friends methods assume a binary social graph
    (see :class:`~components.socialgraph.BinarySocialGraph`).

    Item attributes are represented by a `|U|x|I|` matrix, where `|I|` is the
    number of items in the system. For each item `i` and user `u`, we define a
    score that determines the interactions `u` had with `i`. Again, this could
    just be a binary relationship.


    Parameters
    -----------

        num_users: int (optional, default: 100)
            The number of users `|U|` in the system.

        num_items: int (optional, default: 1250)
            The number of items `|I|` in the system.

        item_representation: :obj:`numpy.ndarray` or None (optional, default: None)
            A `|U|x|I|` matrix representing the past user interactions. If this
            is not None, `num_items` is ignored.

        user_representation: :obj:`numpy.ndarray` or None (optional, default: None)
            A `|U|x|U|` adjacency matrix representing each users' social network.
            If this is not None, num_users is ignored.

        actual_user_representation: :obj:`numpy.ndarray` or None (optional, default: None)
            A `|U|x|I|` matrix representing the real user scores. This matrix is
            **not** used for recommendations. This is only kept for measurements
            and the system is unaware of it.

        verbose: bool (optional, default: False)
            If True, enables verbose mode. Disabled by default.

        num_items_per_iter: int (optional, default: 10)
            Number of items presented to the user per iteration.

        seed: int, None (optional, default: None)
            Seed for random generator.

    Attributes
    -----------
        Inherited by BaseRecommender : :class:`~models.recommender.BaseRecommender`

    Examples
    ----------
        SocialFiltering can be instantiated with no arguments -- in which case,
        it will be initialized with the default parameters and the item/user
        representation will be initialized to zero. This means that a user
        starts with no followers/users they follow, and that there have been no
        previous interactions for this set of users.

        >>> sf = SocialFiltering()
        >>> sf.users_hat.shape
        (100, 100)   # <-- 100 users (default)
        >>> sf.items.shape
        (100, 1250) # <-- 100 users (default), 1250 items (default)

        This class can be customized either by defining the number of users
        and/or items in the system:

        >>> sf = SocialFiltering(num_users=1200, num_items=5000)
        >>> sf.items.shape
        (1200, 5000) # <-- 1200 users, 5000 items

        >>> sf = ContentFiltering(num_users=50)
        >>> sf.items.shape
        (50, 1250) # <-- 50 users, 1250 items (default)

        Or by generating representations for items and/or users. In the example
        below, items are uniformly distributed. We "indirectly" define 100 users
        by defining a `100x200` item representation.

        >>> item_representation = np.random.randint(2, size=(100, 200))
        # Social networks are drawn from a binomial distribution.
        # This representation also uses 100 users.
        >>> sf = SocialFiltering(item_representation=item_representation)
        >>> sf.items.shape
        (100, 200)
        >>> sf.users_hat.shape
        (100, 100)

        Note that user and item representations have the precedence over the
        number of users/items specified at initialization. For example:

        >>> sf = SocialFiltering(num_users=50, user_representation=user_representation)
        >>> sf.items.shape
        (100, 200) # <-- 100 users, 200 items.
        # Note thatnum_users was ignored because user_representation was specified.

        The same is true about the number of items or users and item representations.

        >>> sf = SocialFiltering(num_users=1400, item_representation=item_representation)
        >>> sf.items.shape
        (100, 200) # <-- 100 attributes, 200 items. num_users was ignored.
        >>> cf.user_profile.shape
        (100, 100) # <-- 100 users (as implicitly specified by item_representation)

        """

    def __init__(  # pylint: disable=too-many-arguments,super-init-not-called
        self,
        num_users=100,
        num_items=1250,
        item_representation=None,
        user_representation=None,
        actual_user_scores=None,
        actual_item_representation=None,
        probabilistic_recommendations=False,
        verbose=False,
        num_items_per_iter=10,
        seed=None,
        **kwargs
    ):
        # Give precedence to user_representation, otherwise build empty one

        if all_none(user_representation, num_users):
            raise ValueError("user_representation and num_users can't be all None")
        if all_none(item_representation, num_items):
            raise ValueError("item_representation and num_items can't be all None")

        if not is_array_valid_or_none(user_representation, ndim=2):
            raise ValueError("user_representation is invalid")
        if not is_array_valid_or_none(item_representation, ndim=2):
            raise ValueError("item_representation is not valid")
        num_items = get_first_valid(
            getattr(item_representation, "shape", [None, None])[1], num_items
        )

        num_users = get_first_valid(
            getattr(user_representation, "shape", [None])[0],
            getattr(item_representation, "shape", [None])[0],
            num_users,
        )

        if user_representation is None:
            user_representation = SocialGraphGenerator.generate_random_graph(
                num=num_users, p=0.3, seed=seed, graph_type=nx.fast_gnp_random_graph
            )
        if item_representation is None:
            item_representation = np.zeros((num_users, num_items), dtype=int)
        # if the actual item representation is not specified, we assume
        # that the recommender system's beliefs about the item attributes
        # are the same as the "true" item attributes
        if actual_item_representation is None:
            actual_item_representation = np.copy(item_representation)
        if not all_besides_none_equal(
            getattr(user_representation, "shape", [None])[0],
            getattr(user_representation, "shape", [None, None])[1],
            num_users,
        ):
            raise ValueError("user_representation must be a square matrix")
        if not all_besides_none_equal(
            getattr(user_representation, "shape", [None, None])[1],
            getattr(item_representation, "shape", [None])[0],
            num_users,
        ):
            raise ValueError(
                "user_representation.shape[1] should be the same as "
                + "item_representation.shape[0]"
            )
        if not all_besides_none_equal(
            getattr(item_representation, "shape", [None, None])[1], num_items
        ):
            raise ValueError("item_representation.shape[1] should be the same as " + "num_items")

        measurements = [MSEMeasurement()]
        # Initialize recommender system
        BaseRecommender.__init__(
            self,
            user_representation,
            item_representation,
            actual_user_scores,
            actual_item_representation,
            num_users,
            num_items,
            num_items_per_iter,
            probabilistic_recommendations=False,
            seed=seed,
            measurements=measurements,
            verbose=verbose,
            **kwargs
        )

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
        interactions_per_user[self.users.user_vector, interactions] = 1
        assert interactions_per_user.shape == self.items_hat.shape
        self.items_hat[:, :] = np.add(self.items_hat, interactions_per_user)
