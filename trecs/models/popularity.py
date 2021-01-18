"""
Popularity-based recommender system
"""
import numpy as np
from trecs.metrics import MSEMeasurement
from trecs.validate import validate_user_item_inputs
from .recommender import BaseRecommender


class PopularityRecommender(BaseRecommender):
    """
    A customizable popularity recommendation system.

    With the popularity recommender system, users are presented items that are
    popular in the system. The popularity of an item is measured by the number
    of times users interacted with that item in the past. In this
    implementation, items do not expire and, therefore, the system does not base
    its choice on how recent the items are.

    Item attributes are represented by a :math:`1\\times|I|` array, where
    :math:`|I|` is the number of items in the system. This array stores the
    number of user interactions for each item.

    User profiles are represented by a :math:`|U|\\times 1` matrix, where
    :math:`|U|` is the number of users in the system. All elements of this matrix
    are equal to 1, as the predictions of the system are solely based on the
    item attributes.

    Parameters
    -----------

        num_users: int (optional, default: 100)
            The number of users :math:`|U|` in the system.

        num_items: int (optional, default: 1250)
            The number of items :math:`|I|` in the system.

        item_representation: :obj:`numpy.ndarray` or None (optional, default: None)
            A :math:`|A|\\times|I|` matrix representing the similarity between
            each item and attribute. If this is not None, `num_items` is ignored.

        user_representation: :obj:`numpy.ndarray` or None (optional, default: None)
            A :math:`|U|\\times|A|` matrix representing the similarity between
            each item and attribute, as interpreted by the system. If this is not
            None, `num_users` is ignored.

        actual_user_representation: :obj:`numpy.ndarray` or None or \
                            :class:`~components.users.Users` (optional, default: None)
            Either a :math:`|U|\\times|T|` matrix representing the real user
            profiles, where :math:`T` is the number of attributes in the real
            underlying user profile, or a `Users` object that contains the real
            user profiles or real user-item scores. This matrix is **not** used
            for recommendations. This is only kept for measurements and the
            system is unaware of it.

        actual_item_representation: :obj:`numpy.ndarray` or None (optional, default: None)
            A :math:`|T|\\times|I|` matrix representing the real user profiles, where
            :math:`T` is the number of attributes in the real underlying item profile.
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

        Note that all arguments passed in at initialization must be consistent -
        otherwise, an error is thrown. For example, one cannot pass in
        `num_users=200` but have `user_representation.shape` be `(300, 1)`.
        Likewise, one cannot pass in `num_items=1000` but have
        `item_representation.shape` be `(1, 500)`.

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
        seed=None,
        verbose=False,
        num_items_per_iter=10,
        **kwargs
    ):
        num_users, num_items, num_attributes = validate_user_item_inputs(
            num_users,
            num_items,
            user_representation,
            item_representation,
            actual_user_representation,
            actual_item_representation,
            100,
            1250,
            num_attributes=1,
        )
        # num_attributes should always be 1
        if item_representation is None:
            item_representation = np.zeros((num_attributes, num_items), dtype=int)
        # if the actual item representation is not specified, we assume
        # that the recommender system's beliefs about the item attributes
        # are the same as the "true" item attributes
        if actual_item_representation is None:
            actual_item_representation = np.copy(item_representation)
        if user_representation is None:
            user_representation = np.ones((num_users, num_attributes), dtype=int)

        measurements = [MSEMeasurement()]

        super().__init__(
            user_representation,
            item_representation,
            actual_user_representation,
            actual_item_representation,
            num_users,
            num_items,
            num_items_per_iter,
            probabilistic_recommendations=probabilistic_recommendations,
            measurements=measurements,
            verbose=verbose,
            seed=seed,
            **kwargs
        )

    def _update_internal_state(self, interactions):
        histogram = np.zeros(self.num_items)
        np.add.at(histogram, interactions, 1)
        self.items_hat[:, :] = np.add(self.items_hat, histogram)

    def process_new_items(self, new_items):
        """
        The popularity of any new items is always zero.

        Parameters:
        ------------
            new_items: numpy.ndarray
                An array of items that represents new items that are being
                added into the system. Should be :math:`|A|\\times|I|`
        """
        # start popularity of new items as 0
        new_representation = np.zeros(new_items.shape[1]).reshape(1, -1)
        self.items_hat = np.hstack([self.items_hat, new_representation])
