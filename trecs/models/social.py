"""
Social filtering recommender system, where users get shown items that were
interacted with by users in their social networks
"""
import networkx as nx
import numpy as np
from trecs.metrics import MSEMeasurement
from trecs.components import BinarySocialGraph
from trecs.random import SocialGraphGenerator
from trecs.utils import get_first_valid, non_none_values, validate_user_item_inputs
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

        actual_user_representation: :obj:`numpy.ndarray` or None or \
                            :class:`~components.users.Users` (optional, default: None)
            Either a `|U|x|T|` matrix representing the real user profiles, where
            `T` is the number of attributes in the real underlying user profile,
            or a `Users` object that contains the real user profiles or real
            user-item scores. This matrix is **not** used for recommendations. This
            is only kept for measurements and the system is unaware of it.

        actual_item_representation: :obj:`numpy.ndarray` or None (optional, default: None)
            A `|T|x|I|` matrix representing the real user profiles, where
            `T` is the number of attributes in the real underlying item profile.
            This matrix is **not** used for recommendations. This
            is only kept for measurements and the system is unaware of it.

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

        Note that all arguments passed in at initialization must be consistent -
        otherwise, an error is thrown. For example, one cannot pass in
        `num_users=200` but have `user_representation.shape` be `(200, 500)` or
        `(300, 300)`. Likewise, one cannot pass in `num_items=1000` but have
        `item_representation.shape` be `(200, 500)`.
        """

    def __init__(  # pylint: disable-all
        self,
        num_users=None,
        num_items=None,
        user_representation=None,
        item_representation=None,
        actual_user_representation=None,
        actual_item_representation=None,
        probabilistic_recommendations=False,
        verbose=False,
        num_items_per_iter=10,
        seed=None,
        **kwargs
    ):
        num_users, num_items = validate_user_item_inputs(
            num_users,
            num_items,
            user_representation,
            item_representation,
            actual_user_representation,
            actual_item_representation,
            None,  # see if we can get the default number of users from the items array
            1250,
        )
        if num_users is None:
            # get user representation from items instead
            num_users = get_first_valid(
                getattr(item_representation, "shape", [None])[0], 100  # default to 100 users
            )

        if user_representation is None:
            user_representation = SocialGraphGenerator.generate_random_graph(
                num=num_users, p=0.3, seed=seed, graph_type=nx.fast_gnp_random_graph
            )
        if item_representation is None:
            item_representation = np.zeros((num_users, num_items), dtype=int)

        # verify that the user representation is an adjacency matrix and that
        # the item representation aligns
        users_vals = non_none_values(
            getattr(user_representation, "shape", [None, None])[1],
            getattr(item_representation, "shape", [None])[0],
            num_users,
        )
        if len(users_vals) > 1:
            raise ValueError("Number of users must be consistent across all inputs")
        # if the actual item representation is not specified, we assume
        # that the recommender system's beliefs about the item attributes
        # are the same as the "true" item attributes
        if actual_item_representation is None:
            actual_item_representation = np.copy(item_representation)

        measurements = [MSEMeasurement()]
        # Initialize recommender system
        BaseRecommender.__init__(
            self,
            user_representation,
            item_representation,
            actual_user_representation,
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
