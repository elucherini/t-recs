import numpy as np
from rec.models import BaseRecommender
from rec.metrics import MSEMeasurement
from rec.utils import (
    get_first_valid,
    is_array_valid_or_none,
    is_equal_dim_or_none,
    all_none,
    is_valid_or_none,
)


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
        >>> pr.user_profiles.shape
        (100, 1)   # <-- 100 users (default)
        >>> pr.item_attributes.shape
        (1, 1250) # <-- 1250 items (default)

        This class can be customized by defining the number of users and/or items
        in the system.

        >>> pr = PopularityRecommender(num_users=1200, num_items=5000)
        >>> pr.user_profiles.shape
        (1200, 1) # <-- 1200 users
        >>> cf.item_attributes.shape
        (1, 5000)

        Or by generating representations for items (user representation can
        also be defined, but they should always be set to all ones). In the
        example below, items are uniformly distributed and have had between 0
        and 10 interactions each.

            >>> item_representation = np.random.randint(11, size=(1, 200))
            >>> pr = PopularityRecommender(item_representation=item_representation)
            >>> cf.item_attributes.shape
            (1, 200)
            >>> cf.user_profiles.shape
            (100, 1)

        Note that user and item representations have the precedence over the
        number of users and the number of items specified at initialization.
        For example:

        >>> user_representation = np.ones((3000, 1))
        >>> pr = PopularityRecommender(num_users=50, user_representation=user_representation)
        >>> pr.user_profiles.shape
        (3000, 1) # <-- 30000 users. num_users was ignored because user_representation was specified.

    """

    def __init__(
        self,
        num_users=100,
        num_items=1250,
        item_representation=None,
        user_representation=None,
        actual_user_representation=None,
        seed=None,
        verbose=False,
        num_items_per_iter=10,
    ):

        if all_none(item_representation, num_items):
            raise ValueError("num_items and item_representation can't be all None")
        if all_none(user_representation, num_users):
            raise ValueError("num_users and user_representation can't be all None")

        if not is_array_valid_or_none(item_representation, ndim=2):
            raise ValueError("item_representation is not valid")
        if not is_array_valid_or_none(user_representation, ndim=2):
            raise ValueError("item_representation is not valid")

        num_items = get_first_valid(
            getattr(item_representation, "shape", [None, None])[1], num_items
        )

        num_users = get_first_valid(
            getattr(user_representation, "shape", [None])[0], num_users
        )

        if item_representation is None:
            item_representation = np.zeros((1, num_items), dtype=int)
        if user_representation is None:
            user_representation = np.ones((num_users, 1), dtype=int)

        if not is_equal_dim_or_none(
            getattr(user_representation, "shape", [None, None])[1],
            getattr(item_representation, "shape", [None])[0],
        ):
            raise ValueError(
                "user_representation.shape[1] should be the same as "
                + "item_representation.shape[0]"
            )
        if not is_equal_dim_or_none(
            getattr(user_representation, "shape", [None])[0], num_users
        ):
            raise ValueError(
                "user_representation.shape[0] should be the same as " + "num_users"
            )
        if not is_equal_dim_or_none(
            getattr(item_representation, "shape", [None, None])[1], num_items
        ):
            raise ValueError(
                "item_representation.shape[1] should be the same as " + "num_items"
            )

        measurements = [MSEMeasurement()]

        super().__init__(
            user_representation,
            item_representation,
            actual_user_representation,
            num_users,
            num_items,
            num_items_per_iter,
            measurements=measurements,
            verbose=verbose,
            seed=seed,
        )

    def _update_user_profiles(self, interactions):
        histogram = np.zeros(self.num_items)
        np.add.at(histogram, interactions, 1)
        self.item_attributes[:, :] = np.add(self.item_attributes, histogram)
