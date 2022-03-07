"""
The content filtering recommender system attempts to match users to items
based on highest predicted inner product between the predicted user profile
and predicted item profile. The predictions of user and item profiles
are generated iteratively as users interact with items.
"""
import numpy as np
from scipy.optimize import nnls
import scipy.sparse as sp
import trecs.matrix_ops as mo
from trecs.random import Generator
from trecs.validate import validate_user_item_inputs
from .recommender import BaseRecommender


class ContentFiltering(BaseRecommender):
    """
    A customizable content-filtering recommendation system.

    With content filtering, items and users are represented by a set of
    attributes A. This class assumes that the attributes used for items and
    users are the same. The recommendation system matches users to items with
    similar attributes.

    Item attributes are represented by a :math:`|A|\\times|I|` matrix, where
    :math:`|I|` is the number of items in the system. For each item, we define
    the similarity to each attribute.

    User profiles are represented by a :math:`|U|\\times|A|` matrix, where
    :math:`|U|` is the number of users in the system. For each user, we define
    the similarity to each attribute.

    Parameters
    -----------

        num_users: int, default 100
            The number of users :math:`|U|` in the system.

        num_items: int, default 1250
            The number of items :math:`|I|` in the system.

        num_attributes: int, default 1000
            The number of attributes :math:`|A|` in the system.

        user_representation: :obj:`numpy.ndarray`, optional
            A :math:`|U|\\times|A|` matrix representing the similarity between
            each item and attribute, as interpreted by the system.

        item_representation: :obj:`numpy.ndarray`, optional
            A :math:`|A|\\times|I|` matrix representing the similarity between
            each item and attribute.

        actual_user_representation: :obj:`numpy.ndarray` or \
                            :class:`~components.users.Users`, optional
            Either a :math:`|U|\\times|T|` matrix representing the real user profiles, where
            :math:`T` is the number of attributes in the real underlying user profile,
            or a `Users` object that contains the real user profiles or real
            user-item scores. This matrix is **not** used for recommendations. This
            is only kept for measurements and the system is unaware of it.

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
        ContentFiltering can be instantiated with no arguments -- in which case,
        it will be initialized with the default parameters and the item/user
        representations will be assigned randomly.

        >>> cf = ContentFiltering()
        >>> cf.users_hat.shape
        (100, 1000)   # <-- 100 users (default), 1000 attributes (default)
        >>> cf.items.shape
        (1000, 1250) # <-- 1000 attributes (default), 1250 items (default)

        This class can be customized either by defining the number of users/items/attributes
        in the system.

        >>> cf = ContentFiltering(num_users=1200, num_items=5000)
        >>> cf.users_hat.shape
        (1200, 1000) # <-- 1200 users, 1000 attributes

        >>> cf = ContentFiltering(num_users=1200, num_items=5000, num_attributes=2000)
        >>> cf.users_hat.shape
        (1200, 2000) # <-- 1200 users, 2000 attributes

        Or by generating representations for items and/or users. In the example
        below, items are uniformly distributed. We indirectly define 100
        attributes by defining the following ``item_representation``:

        >>> items = np.random.randint(0, 1, size=(100, 200))
        # Users are represented by a power law distribution.
        # This representation also uses 100 attributes.
        >>> power_dist = Distribution(distr_type='powerlaw')
        >>> users = power_dist.compute(a=1.16, size=(30, 100)).compute()
        >>> cf = ContentFiltering(item_representation=items, user_representation=users)
        >>> cf.items.shape
        (100, 200)
        >>> cf.users_hat.shape
        (30, 100)

        Note that all arguments passed in at initialization must be consistent -
        otherwise, an error is thrown. For example, one cannot pass in
        ``num_users=200`` but have ``user_representation.shape`` be ``(300, 100)``.
        Likewise, one cannot pass in ``num_items=1000`` but have
        ``item_representation.shape`` be ``(100, 500)``.
    """

    def __init__(  # pylint: disable-all
        self,
        num_users=None,
        num_items=None,
        num_attributes=None,
        user_representation=None,
        item_representation=None,
        actual_user_representation=None,
        actual_item_representation=None,
        probabilistic_recommendations=False,
        seed=None,
        num_items_per_iter=10,
        **kwargs
    ):
        # pylint: disable=duplicate-code
        num_users, num_items, num_attributes = validate_user_item_inputs(
            num_users,
            num_items,
            user_representation,
            item_representation,
            actual_user_representation,
            actual_item_representation,
            100,
            1250,
            1000,
            num_attributes=num_attributes,
        )

        # generate recommender's initial "beliefs" about user profiles
        # and item attributes
        if user_representation is None:
            user_representation = np.zeros((num_users, num_attributes))
        if item_representation is None:
            item_representation = Generator(seed=seed).binomial(
                n=1, p=0.5, size=(num_attributes, num_items)
            )
        # if the actual item representation is not specified, we assume
        # that the recommender system's beliefs about the item attributes
        # are the same as the "true" item attributes
        if actual_item_representation is None:
            actual_item_representation = item_representation.copy()

        # initialize cumulative interactions as a sparse matrix
        self.all_interactions = None

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
            probabilistic_recommendations=probabilistic_recommendations,
            seed=seed,
            **kwargs
        )

        # set cumulative interactions as a sparse matrix
        self.all_interactions = sp.csr_matrix((self.num_users, self.num_items), dtype=int)

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
            interactions: :obj:`numpy.ndarray`
                An array of item indices that users have interacted with in the
                latest step. Namely, `interactions[u]` represents the index of
                the item that the user has interacted with.

        """
        sparse_interactions = sp.csr_matrix(
            (np.ones(interactions.shape), (self.users.user_vector, interactions)),
            self.all_interactions.shape,
        )
        self.all_interactions = self.all_interactions + sparse_interactions

    def train(self):
        """
        Uses the NNLS solver to train the user representations, based on the user
        interaction & item attribute data.

        Note: this function may run slowly because it requires a manual loop over every
        user.
        """
        if (
            self.all_interactions is not None and self.all_interactions.sum() > 0
        ):  # if there are interactions present:
            for i in range(self.num_users):
                item_attr = mo.to_dense(
                    self.predicted_item_attributes.T
                )  # convert to dense so nnls can be used
                user_interactions = self.all_interactions[i, :].toarray()[0, :]
                # solve for Content Filtering representation using nnls solver
                self.users_hat.value[i, :] = nnls(item_attr, user_interactions)[0]

        super().train()

    def process_new_items(self, new_items):
        """
        We assume the content filtering system has perfect knowledge
        of the new items; therefore, when new items are created,
        we simply return the new item attributes.

        Parameters
        ------------
            new_items: :obj:`numpy.ndarray`
                An array of items that represents new items that are being
                added into the system. Should be :math:`|A|\\times|I|`
        """
        # add indices for new items into all interactions matrix
        empty_interactions = sp.csr_matrix((self.num_users, new_items.shape[1]), dtype=int)
        self.all_interactions = sp.hstack([self.all_interactions, empty_interactions])
        return new_items

    def process_new_users(self, new_users, **kwargs):
        """
        By default, the content filtering system assumes the predicted user profiles
        are zero vectors. (Note that this effectively corresponds to providing
        random recommendations to each user).

        Parameters
        ------------
           new_users: :obj:`numpy.ndarray`
                An array of users that represents new users that are being
                added into the system. Should be of dimension :math:`|U|\\times|A|`
        """
        # add indices for new items into all interactions matrix
        num_new = new_users.shape[0]
        empty_interactions = sp.csr_matrix((num_new, self.num_items), dtype=int)
        self.all_interactions = sp.vstack([self.all_interactions, empty_interactions])
        # each user is initially represented as zeros
        return np.zeros((num_new, self.users_hat.num_attrs))
