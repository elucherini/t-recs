"""
Popularity-based recommender system
"""
import numpy as np
from rec.metrics import MSEMeasurement
from rec.utils import validate_user_item_inputs
from .recommender import BaseRecommender


class PopularityRecommender(BaseRecommender):
    """
    A customizable popularity recommendation system.

    With the popularity recommender system, users are presented items that are
    popular in the system. The popularity of an item is measured by the number
    of times users interacted with that item in the past. In this
    implementation, items do not expire and, therefore, the system does not base
    its choice on how recent the items are.

    Item attributes are represented by a `1x|I|` array, where `|I|` is the
    number of items in the system. This array stores the number of user
    interactions for each item.

    User profiles are represented by a `|U|x1` matrix, where `|U|` is the number
    of users in the system. All elements of this matrix are equal to 1, as the
    predictions of the system are solely based on the item attributes.

    Parameters
    -----------

        num_users: int (optional, default: 100)
            The number of users `|U|` in the system.

        num_items: int (optional, default: 1250)
            The number of items `|I|` in the system.

        item_representation: :obj:`numpy.ndarray` or None (optional, default: None)
            A `|A|x|I|` matrix representing the similarity between each item
            and attribute. If this is not None, num_items is ignored.

        user_representation: :obj:`numpy.ndarray` or None (optional, default: None)
            A `|U|x|A|` matrix representing the similarity between each item
            and attribute, as interpreted by the system. If this is not None, num_users is ignored.

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
        Inherited by BaseRecommender: :class:`~models.recommender.BaseRecommender`

    Examples
    ---------
        PopularityRecommender can be instantiated with no arguments -- in which
        case, it will be initialized with the default parameters.

        >>> pr = PopularityRecommender()
        >>> pr.users_hat.shape
        (100, 1)   # <-- 100 users (default)
        >>> pr.items.shape
        (1, 1250) # <-- 1250 items (default)

        This class can be customized by defining the number of users and/or items
        in the system.

        >>> pr = PopularityRecommender(num_users=1200, num_items=5000)
        >>> pr.users_hat.shape
        (1200, 1) # <-- 1200 users
        >>> cf.items.shape
        (1, 5000)

        Or by generating representations for items (user representation can
        also be defined, but they should always be set to all ones). In the
        example below, items are uniformly distributed and have had between 0
        and 10 interactions each.

            >>> item_representation = np.random.randint(11, size=(1, 200))
            >>> pr = PopularityRecommender(item_representation=item_representation)
            >>> cf.items.shape
            (1, 200)
            >>> cf.users_hat.shape
            (100, 1)

        Note that user and item representations have the precedence over the
        number of users and the number of items specified at initialization.
        For example:

        >>> user_representation = np.ones((3000, 1))
        >>> pr = PopularityRecommender(num_users=50, user_representation=user_representation)
        >>> pr.users_hat.shape
        (3000, 1)
        # 30000 users. num_users was ignored because user_representation was specified.

    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        num_users=None,
        num_items=None,
        item_representation=None,
        user_representation=None,
        actual_user_representation=None,
        actual_item_representation=None,
        probabilistic_recommendations=False,
        seed=None,
        verbose=False,
        num_items_per_iter=10,
        **kwargs
    ):
        num_users, num_items = validate_user_item_inputs(
            num_users,
            num_items,
            item_representation,
            user_representation,
            actual_item_representation,
            actual_user_representation,
            100,
            1250,
        )

        if item_representation is None:
            item_representation = np.zeros((1, num_items), dtype=int)
        # if the actual item representation is not specified, we assume
        # that the recommender system's beliefs about the item attributes
        # are the same as the "true" item attributes
        if actual_item_representation is None:
            actual_item_representation = np.copy(item_representation)
        if user_representation is None:
            user_representation = np.ones((num_users, 1), dtype=int)

        measurements = [MSEMeasurement()]

        super().__init__(
            user_representation,
            item_representation,
            actual_user_representation,
            actual_item_representation,
            num_users,
            num_items,
            num_items_per_iter,
            probabilistic_recommendations=False,
            measurements=measurements,
            verbose=verbose,
            seed=seed,
            **kwargs
        )

    def _update_user_profiles(self, interactions):
        histogram = np.zeros(self.num_items)
        np.add.at(histogram, interactions, 1)
        self.items_hat[:, :] = np.add(self.items_hat, histogram)
