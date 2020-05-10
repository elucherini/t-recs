
from .measurement import MSEMeasurement, HomogeneityMeasurement
import numpy as np
from .recommender import Recommender
from .distribution import Generator
from .utils import get_first_valid, is_array_valid_or_none, is_equal_dim_or_none, all_none, is_valid_or_none

class ContentFiltering(Recommender):
    """A customizable content-filtering recommendation system.

    With content filtering, items and users are represented by a set of attributes A. This class
    assumes that the attributes used for items and users are the same. The recommendation system
    matches users to items with similar attributes.

    Item attributes are represented by a |A|x|I| matrix, where |I| is the number of items in the system.
    For each item, we define the similarity to each attribute.

    User profiles are represented by a |U|x|A| matrix, where |U| is the number of users in the system.
    For each user, we define the similarity to each attribute.

    Args:
        num_users (int, optional): The number of users |U| in the system.
        num_items (int, optiona;): The number of items |I| in the system.
        num_attributes (int, optional): The number of attributes |A| in the system.
        item_representation (:obj:`numpy.ndarray`, optional): A |A|x|I| matrix representing the similarity
            between each item and attribute. If this is not None, num_items is
            ignored.
        user_representation (:obj:`numpy.ndarray`, optional): A |U|x|A| matrix representing the similarity
            between each item and attribute, as interpreted by the system. If this
            is not None, num_users is ignored.
        actual_user_representation (:obj:`numpy.ndarray`, optional): A |U|x|I| matrix representing the real
            user scores. This matrix is *not* used for recommendations. This is
            only kept for measurements and the system is unaware of it.
        verbose (bool, optional): If True, enables verbose mode. Disabled by default.
        num_items_per_iter (int, optional): Number of items presented to the user per iteration.
        num_new_items (int, optional): Number of new items that the systems add if it runs out
            of items that the user can interact with.

    Attributes:
        Inherited by :class:`Recommender`

    Examples:
        ContentFiltering can be instantiated with no arguments -- in which case, it will
        be initialized with the default parameters and the number of attributes and the
        item/user representations will be assigned randomly.

        >>> cf = ContentFiltering()
        >>> cf.user_profiles.shape
        (100, 99)   # <-- 100 users (default), 99 attributes (randomly generated)
        >>> cf.item_attributes.shape
        (99, 1250) # <-- 99 attributes (randomly generated), 1250 items (default)

        >>> cf1 = ContentFiltering()
        >>> cf.user_profiles.shape
        (100, 582) # <-- 100 users (default), 582 attributes (randomly generated)

        This class can be customized either by defining the number of users/items
        in the system. The number of attributes will still be random, unless specified.

        >>> cf = ContentFiltering(num_users=1200, num_items=5000)
        >>> cf.user_profiles.shape
        (1200, 2341) # <-- 1200 users, 2341 attributes

        >>> cf = ContentFiltering(num_users=1200, num_items=5000, num_attributes=2000)
        >>> cf.user_profiles.shape
        (1200, 2000) # <-- 1200 users, 2000 attributes

        Or by generating representations for items and/or users:
        # Items are uniformly distributed. We indirectly define 100 attributes.
        >>> item_representation = np.random.randint(0, 1, size=(100, 200))
        # Users are represented by a power law distribution. This representation also uses 100 attributes.
        >>> user_representation = Distribution(distr_type='powerlaw').compute(a=1.16, size=(30, 100)).compute()
        >>> cf = ContentFiltering(item_representation=item_representation, user_representation=user_representation)
        >>> cf.item_attributes.shape
        (100, 200)
        >>> cf.user_profiles.shape
        (30, 100)

        Note that user and item representations have the precedence over the number of
        users/items/attributes specified at initialization. For example:
        >>> cf = ContentFiltering(num_users=50, user_representation=user_representation)
        >>> cf.user_profiles.shape
        (30, 100) # <-- 30 users, 100 attributes. num_users was ignored because user_representation was specified.

        The same happens with the number of items and the number of attributes.
        In the latter case, the explicit number of attributes is ignored:

        >>> cf = ContentFiltering(num_attributes=1400, item_representation=item_representation)
        >>> cf.item_attributes.shape
        (100, 200) # <-- 100 attributes, 200 items. num_attributes was ignored.
        >>> cf.user_profiles.shape
        (100, 100) # <-- 100 users (default), 100 attributes (as implicitly specified by item_representation)

    """
    def __init__(self, num_users=100, num_items=1250, num_attributes=1000,
        item_representation=None, user_representation=None,
        actual_user_representation=None,
        verbose=False, num_items_per_iter=10, num_new_items=30):

        # Give precedence to item_representation, otherwise build random one
        if all_none(item_representation, num_items):
                raise ValueError("num_items and item_representation can't be all None")
        if all_none(user_representation, num_users):
                raise ValueError("num_users and user_representation can't be all None")
        if all_none(user_representation, item_representation, num_attributes):
                raise ValueError("item_representation, user_representation, and " + \
                                 "num_attributes can't be all None")

        if not is_array_valid_or_none(item_representation, ndim=2, square=False):
            raise ValueError("item_representation is not valid")
        if not is_array_valid_or_none(user_representation, ndim=2, square=False):
            raise ValueError("item_representation is not valid")
        if not is_valid_or_none(num_attributes, int):
            raise TypeError("num_attributes must be an int")
        if not is_equal_dim_or_none(getattr(user_representation,
                                            'shape', [None, None])[1],
                                    getattr(item_representation,
                                            'shape', [None])[0]):
            raise ValueError("user_representation.shape[1] should be the same as " + \
                             "item_representation.shape[0]")
        num_items = get_first_valid(getattr(item_representation, 'shape',
                                            [None, None])[1],
                                    num_items)
        num_attributes = get_first_valid(getattr(item_representation, 'shape',
                                                 [None])[0],
                                        getattr(user_representation, 'shape',
                                                [None, None])[1],
                                        num_attributes)
        if item_representation is None:
            item_representation = Generator().binomial(n=.3, p=1,
                                                      size=(num_attributes, num_items))

        assert(num_attributes is not None)
        assert(item_representation is not None)
        self.num_attributes = num_attributes
        # Give precedence to user_representation, otherwise build random one
        num_users = get_first_valid(getattr(user_representation, 'shape', [None])[0],
                                    num_users)
        if user_representation is None:
            user_representation = np.zeros((num_users, num_attributes))

        assert(user_representation is not None)
        self.measurements = [MSEMeasurement(), HomogeneityMeasurement()]

        # Initialize recommender system
        Recommender.__init__(self, user_representation, item_representation,
                             actual_user_representation, num_users, num_items,
                             num_items_per_iter, num_new_items, verbose=verbose)

    def _update_user_profiles(self, interactions):
        """ Private function that updates user profiles with data from
            latest interactions.

            Specifically, this function converts interactions into attributes.
            For example, if user u has interacted with an item that has attributes
            a1 and a2, user u's profile will be updated by increasing the similarity
            to attributes a1 and a2.

        Args:
            interactions (numpy.ndarray): An array of item indices that users have
                interacted with in the latest step. Namely, interactions_u represents
                the index of the item that the user has interacted with.

        """
        interactions_per_user = np.zeros((self.num_users, self.num_items))
        interactions_per_user[self.user_vector, interactions] = 1
        user_attributes = np.dot(interactions_per_user, self.item_attributes.T)
        self.user_profiles = np.add(self.user_profiles, user_attributes)

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
        assert(self.user_profiles.shape[1] == self.item_attributes.shape[0])
        return Recommender.train(self, user_profiles, item_attributes, normalize=normalize)

