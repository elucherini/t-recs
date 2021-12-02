"""
Random recommendation system, where users are shown items that are uniformly randomly
sampled from the full item catalog
"""
import numpy as np
import warnings

from trecs.validate import validate_user_item_inputs
from .recommender import BaseRecommender


class RandomRecommender(BaseRecommender):
    """
    A recommender system that randomly recommends items to each user.

    Item attributes are represented by a :math:`1\\times|I|` array, where
    :math:`|I|` is the number of items in the system. This array stores the
    number of user interactions for each item.

    User profiles are represented by a :math:`|U|\\times 1` matrix, where
    :math:`|U|` is the number of users in the system. All elements of this matrix
    are equal to 1, as the predictions of the system are solely based on the
    item attributes.

    Parameters
    -----------

        num_users: int, default 100
            The number of users :math:`|U|` in the system.

        num_items: int, default 1250
            The number of items :math:`|I|` in the system.

        actual_user_representation: :obj:`numpy.ndarray` or \
                            :class:`~components.users.Users`, optional
            Either a :math:`|U|\\times|T|` matrix representing the real user
            profiles, where :math:`T` is the number of attributes in the real
            underlying user profile, or a `Users` object that contains the real
            user profiles or real user-item scores. This matrix is **not** used
            for recommendations. This is only kept for measurements and the
            system is unaware of it.

        actual_item_representation: :obj:`numpy.ndarray`, optional
            A :math:`|T|\\times|I|` matrix representing the real user profiles, where
            :math:`T` is the number of attributes in the real underlying item profile.
            This matrix is **not** used for recommendations. This
            is only kept for measurements and the system is unaware of it.

        num_items_per_iter: int, default 10
            Number of items presented to the user per iteration.

        seed: int, optional
            Seed for random generator.

    Attributes
    -----------
        Inherited from BaseRecommender: :class:`~models.recommender.BaseRecommender`

    Examples
    ---------
        RandomRecommender can be instantiated with no arguments -- in which
        case, it will be initialized with the default parameters.

        >>> rr = RandomRecommender()
        >>> rr.users_hat.shape
        (100, 1)   # <-- 100 users (default)
        >>> rr.items.shape
        (1, 1250) # <-- 1250 items (default)

        This class can be customized by defining the number of users and/or items
        in the system.

        >>> rr = RandomRecommender(num_users=1200, num_items=5000)
        >>> rr.users_hat.shape
        (1200, 1) # <-- 1200 users
        >>> rr.items.shape
        (1, 5000)

        If the arguments `score_fn`, `user_representation`, or `item_representation`
        are passed in, they will be ignored, since the random recommender relies on
        these representations. Note that all arguments passed in at initialization
        must be consistent - otherwise, an error is thrown.

    """

    def __init__(  # pylint: disable-all
        self,
        num_users=None,
        num_items=None,
        actual_user_representation=None,
        actual_item_representation=None,
        num_items_per_iter=10,
        **kwargs
    ):
        if kwargs.pop("score_fn", None) is not None:
            warnings.warn(
                "score_fn cannot be passed to RandomRecommender; user-item scores must be generated randomly."
            )
        if kwargs.pop("user_representation", None) is not None:
            warnings.warn(
                "user_representation is not relevant for the RandomRecommender; overwriting user_representation."
            )
        if kwargs.pop("item_representation", None) is not None:
            warnings.warn(
                "item_representation is not relevant for the RandomRecommender; overwriting item_representation."
            )

        num_users, num_items = validate_user_item_inputs(
            num_users,
            num_items,
            None,
            None,
            actual_user_representation,
            actual_item_representation,
            100,
            1250,
            # attributes are guaranteed to match due to
            # the fact that we set the item / user representations
            attributes_must_match=False,
        )
        # items and users will always be zeros; this ensures recommendation in
        # random order due to the "tiebreaking" functionality in random.py
        item_representation = np.zeros((1, num_items), dtype=int)
        user_representation = np.zeros((num_users, 1), dtype=int)

        # if the actual item representation is not specified, we assume
        # that the recommender system's beliefs about the item attributes
        # are the same as the "true" item attributes
        if actual_item_representation is None:
            actual_item_representation = item_representation.copy()

        super().__init__(
            user_representation,
            item_representation,
            actual_user_representation,
            actual_item_representation,
            num_users,
            num_items,
            num_items_per_iter,
            **kwargs
        )

    def _update_internal_state(self, interactions):
        """
        Internal representation of items and users does not change based
        on interactions.
        """
        pass

    def process_new_items(self, new_items):
        """
        The representation of any new items is always zero.

        Parameters
        ------------
            new_items: :obj:`numpy.ndarray`
                An array of items that represents new items that are being
                added into the system. Should be :math:`|A|\\times|I|`
        """
        # start popularity of new items as 0
        new_representation = np.zeros(new_items.shape[1]).reshape(1, -1)
        return new_representation

    def process_new_users(self, new_users, **kwargs):
        """
        The representation of any new users is always zero.

        ------------
           new_users: :obj:`numpy.ndarray`
                An array of users that represents new users that are being
                added into the system. Should be of dimension :math:`|U|\\times|A|`
        """
        # start popularity of new users as 0
        new_representation = np.zeros(new_users.shape[0]).reshape(-1, 1)
        return new_representation
