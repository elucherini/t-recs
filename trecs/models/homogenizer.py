""" Content filtering class """
import numpy as np
from trecs.metrics import MSEMeasurement
from trecs.random import Generator
from trecs.utils import (
    non_none_values,
    is_valid_or_none,
)
from trecs.validate import validate_user_item_inputs
from .recommender import BaseRecommender
from .content import ContentFiltering


class Homogenizer(BaseRecommender):
    """
    A customizable recommendation system that takes another model class and forces incremental homogenization of user representation.

    With Homogenizer, items and users are represented by a set of
    attributes A. This class assumes that the attributes used for items and
    users are the same. The Homogenizer class can be used as a comparison
    point for homogenization that "naturally" emerges in another model class.

    Item attributes are represented by a :math:`|A|\\times|I|` matrix, where
    :math:`|I|` is the number of items in the system. For each item, we define
    the similarity to each attribute.

    User profiles are represented by a :math:`|U|\\times|A|` matrix, where
    :math:`|U|` is the number of users in the system. For each user, we define
    the similarity to each attribute.

    Parameters
    -----------

        num_users: int (optional, default: 100)
            The number of users :math:`|U|` in the system.

        num_items: int (optional, default: 1250)
            The number of items :math:`|I|` in the system.

        num_attributes: int (optional, default: 1000)
            The number of attributes :math:`|A|` in the system.

        user_representation: :obj:`numpy.ndarray` or None (optional, default: None)
            A :math:`|U|\\times|A|` matrix representing the similarity between
            each item and attribute, as interpreted by the system.

        item_representation: :obj:`numpy.ndarray` or None (optional, default: None)
            A :math:`|A|\\times|I|` matrix representing the similarity between
            each item and attribute.

        actual_user_representation: :obj:`numpy.ndarray` or None or \
                            :class:`~components.users.Users` (optional, default: None)
            Either a :math:`|U|\\times|T|` matrix representing the real user profiles, where
            :math:`T` is the number of attributes in the real underlying user profile,
            or a `Users` object that contains the real user profiles or real
            user-item scores. This matrix is **not** used for recommendations. This
            is only kept for measurements and the system is unaware of it.

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

        homogenization_increment: float, None (optional, default: 0.1)
            Proportion increase in homogenization for each timestep

    Attributes
    -----------
        Inherited by BaseRecommender: :class:`~models.recommender.BaseRecommender`

    Examples
    ---------
        Homogenizer can be instantiated with no arguments -- in which case,
        it will be initialized with the default parameters and the item/user
        representations will be assigned randomly.

        >>> hm = Homogenizer()
        >>> hm.users_hat.shape
        (100, 10)   # <-- 100 users (default), 10 attributes (default)
        >>> hm.items.shape
        (10, 1250) # <-- 10 attributes (default), 1250 items (default)

        This class can be customized either by defining the number of users/items/attributes
        in the system as well as by the specified model class.

        >>> hm = Homogenizer(num_users=1200, num_items=5000, model_class=ImplicitMF)
        >>> hm.users_hat.shape
        (1200, 10) # <-- 1200 users, 10 attributes

        >>> hm = Homo(num_users=1200, num_items=5000, num_attributes=2000)
        >>> hm.users_hat.shape
        (1200, 2000) # <-- 1200 users, 2000 attributes

        Or by generating representations for items and/or users. In the example
        below, items are uniformly distributed. We indirectly define 100
        attributes by defining the following `item_representation`:
        >>> number_of_users = 5
        >>> number_of_attributes = 10
        >>> number_of_items = 15
        >>> users = np.random.randint(4, size=(number_of_users, number_of_attributes))
        >>> items = Generator().binomial(n=1, p=.3,size=(number_of_attributes, number_of_items))
        >>> hm = Homogenizer(item_representation=items, user_representation=users)
        >>> hm.items.shape
        (10, 15)
        >>> hm.users_hat.shape
        (5, 10)

        Note that all arguments passed in at initialization must be consistent -
        otherwise, an error is thrown. For example, one cannot pass in
        `num_users=200` but have `user_representation.shape` be `(300, 100)`.
        Likewise, one cannot pass in `num_items=1000` but have
        `item_representation.shape` be `(100, 500)`.
    """

    def __init__(  # pylint: disable-all
        self,
        num_users=None,
        num_items=None,
        num_attributes=10,
        user_representation=None,
        item_representation=None,
        actual_user_representation=None,
        actual_item_representation=None,
        probabilistic_recommendations=False,
        seed=None,
        verbose=False,
        num_items_per_iter=10,
        homogenization_increment=0.1,
        model_class=ContentFiltering,
        **kwargs
    ):

        self.__class__ = type(
            self.__class__.__name__, (model_class, object), dict(self.__class__.__dict__)
        )

        # measurements = [MSEMeasurement()]

        super(self.__class__, self).__init__(
            num_users,
            num_items,
            num_attributes,
            user_representation,
            item_representation,
            actual_user_representation,
            actual_item_representation,
            probabilistic_recommendations,
            seed,
            verbose,
            num_items_per_iter,
            **kwargs
        )

        assert 0 < homogenization_increment < 1

        gen = Generator(seed=seed)

        self.homogenized_users_hat = np.tile(
            gen.normal(size=(1, num_attributes)), (self.num_users, 1)
        )
        # attribute that retains the indivdiual users' preference representation in the algorithn
        self.individual_users_hat = self.users_hat
        # proportion of users_hat that will come from the homogenized representation. Will be updated by homogenization_increment
        self.homogenization_proportion = 0
        self.homogenization_increment = homogenization_increment

    def _update_internal_state(self, interactions):
        """
        Private function that updates user profiles with data from latest
        interactions.

        Specifically, this function converts interactions into attributes.
        For example, if user `u` has interacted with an item that has attributes
        `a1` and `a2`, user `u`'s profile will be updated by increasing the
        similarity to attributes `a1` and `a2`.

        Parameters:
        ------------
            interactions: numpy.ndarray
                An array of item indices that users have interacted with in the
                latest step. Namely, `interactions[u]` represents the index of
                the item that the user has interacted with.

        """
        interactions_per_user = np.zeros((self.num_users, self.num_items))
        interactions_per_user[self.users.user_vector, interactions] = 1
        user_attributes = np.dot(interactions_per_user, self.items_hat.T)

        self.individual_users_hat += user_attributes

        if self.homogenization_proportion < 1:
            self.users_hat = (self.homogenization_proportion * self.homogenized_users_hat) + (
                (1 - self.homogenization_proportion) * self.individual_users_hat
            )
            self.homogenization_proportion += self.homogenization_increment
        else:
            self.users_hat = self.homogenized_users_hat

    def process_new_items(self, new_items):
        """
        As in content filtering, the homogenizer system has perfect knowledge
        of the new items; therefore, when new items are created,
        we simply return the new item attributes.

        Parameters:
        ------------
            new_items: numpy.ndarray
                An array of items that represents new items that are being
                added into the system. Should be :math:`|A|\\times|I|`
        """
        self.items_hat = np.hstack([self.items_hat, new_items])
