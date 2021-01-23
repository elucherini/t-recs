"""
Implicit MF recommender system
"""
import numpy as np
from lenskit.algorithms import als
import pandas as pd
import warnings
from trecs.metrics import MSEMeasurement
from trecs.random import Generator
from trecs.validate import validate_user_item_inputs
from trecs.utils import non_none_values
from .recommender import BaseRecommender


class ImplicitMF(BaseRecommender):
    """
    A customizable implicit matrix factorization recommendation system.

    In the implicit matrix factorization model, there is first a training
    period in which users choices are recorded by the recommender system
    in a user-item interaction matrix, where each cell of the matrix represents
    the number of times the given user (row) has interacted with a particular item
    (column). After training data is collected, the MF model performs a "fitting"
    operation to produce latent feature vectors for both the users and the
    items. These latent feature vectors are used as the basis for future
    recommendations. The MF model may be "refit".

    Item attributes are represented by a :math:`k\\times|I|` array, where
    :math:`|I|` is the number of items in the system and :math:`k` is the
    number of features in the latent representation.

    User profiles are represented by a :math:`|U|\\times k` matrix, where
    :math:`|U|` is the number of users in the system.

    See the underlying implementation used from `lkpy`_.

    .. _`lkpy`: https://github.com/lenskit/lkpy

    Parameters
    -----------

        num_users: int (optional, default: 100)
            The number of users :math:`|U|` in the system.

        num_items: int (optional, default: 1250)
            The number of items :math:`|I|` in the system.

        num_latent_factors: int (optional, default: 10)
            The number of latent factors that will be used to fit the Implicit MF
            model.

        item_representation: :obj:`numpy.ndarray` or None (optional, default: None)
            A :math:`|A|\\times|I|` matrix representing the latent representations
            of the items. If this is not None, `num_items` is ignored.

        user_representation: :obj:`numpy.ndarray` or None (optional, default: None)
            A :math:`|U|\\times|A|` matrix representing the latent reprsentations
            of the users. If this is not None, `num_users` is ignored.

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

        model_params: dict (optional, default: None)
            Arguments that can be passed to `lenskit.algorithms.als.ImplicitMF()` that
            dictate model training. For example: `{'iterations':40, 'reg':0.01}`.

        seed: int, None (optional, default: None)
            Seed for random generator.

    Attributes
    -----------
        Inherited by BaseRecommender: :class:`~models.recommender.BaseRecommender`

    Examples
    ---------
        ImplicitMF can be instantiated with no arguments -- in which case,
        it will be initialized with the default parameters and the item/user
        representations will be assigned randomly.

        >>> mf = ImplicitMF()
        >>> mf.users_hat.shape
        (100, 10)   # <-- 100 users (default), 10 latent features (default)
        >>> mf.items.shape
        (10, 1250) # <-- 10 latent features (default), 1250 items (default)

        This class can be customized either by defining the number of users/items/latent
        features in the system.

        >>> mf = ImplicitMF(num_users=1200, num_items=5000)
        >>> mf.users_hat.shape
        (1200, 10) # <-- 1200 users, 10 attributes

        >>> mf = ImplicitMF(num_users=1200, num_items=5000, num_latent_features=2000)
        >>> mf.users_hat.shape
        (1200, 2000) # <-- 1200 users, 2000 attributes

        Note that all arguments passed in at initialization must be consistent -
        otherwise, an error is thrown. For example, one cannot pass in
        `num_users=200` but have `user_representation.shape` be `(300, 100)`.
        Likewise, one cannot pass in `num_items=1000` but have
        `item_representation.shape` be `(100, 500)`.

        To use the model, it's advised to use the `startup_and_train` method, which
        will collect user interaction data from random recommendations. At the end
        of the startup period, an implicit matrix factorization model is fit to the
        interaction data, generating new user/item latent representations. These
        representations are fixed and used for future calls to `run`.

        >>> mf = ImplicitMF(num_latent_features=10) # 10 latent features
        >>> mf.startup_and_train(50) # 50 timesteps of training
        >>> mf.run(100)

        If you'd like to retrain the matrix factorization model at any point, you
        may use the `train` method. It will fit a new set of latent user/item
        representations based on the interaction data from the *most recent* call to
        `run()`.

        >>> mf.run(50)
        >>> mf.run(50) # on new call to `run`, interaction data from previous `run` is discarded
        >>> mf.train() # MF model fit to data from most recent call to `run`

    """

    def __init__(  # pylint: disable-all
        self,
        num_users=None,
        num_items=None,
        num_latent_factors=None,
        user_representation=None,
        item_representation=None,
        actual_user_representation=None,
        actual_item_representation=None,
        probabilistic_recommendations=False,
        seed=None,
        verbose=False,
        num_items_per_iter=10,
        model_params=None,
        **kwargs
    ):
        # check inputs
        num_users, num_items, num_attributes = validate_user_item_inputs(
            num_users,
            num_items,
            user_representation,
            item_representation,
            actual_user_representation,
            actual_item_representation,
            100,
            1250,
            10,
            num_latent_factors,
        )
        if model_params is None:
            model_params = {}
        num_features_vals = non_none_values(
            model_params.get("features"), num_latent_factors, num_attributes
        )
        if len(num_features_vals) > 1:
            raise ValueError("Number of latent factors is not the same across inputs.")

        self.num_latent_factors = num_attributes
        self.model_params = model_params
        self.als_model = None  # initialize empty model
        self.all_interactions = pd.DataFrame(columns=["user", "item"])  # empty interactions matrix

        # generate user and item representations as needed
        # note that these will be overwritten for all users/items with at least
        # one interaction by a future call to `train`
        gen = Generator(seed=seed)
        if user_representation is None:
            user_representation = gen.normal(size=(num_users, num_attributes))
        if item_representation is None:
            item_representation = gen.normal(size=(num_attributes, num_items))
        # if the actual item representation is not specified, we assume
        # that the recommender system's beliefs about the item attributes
        # are the same as the "true" item attributes
        if actual_item_representation is None:
            actual_item_representation = np.copy(item_representation)

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
        """
        At each training timestep, we keep track of the user/item interactions.
        """
        user_item_ids = tuple(zip(self.users.user_vector, interactions))
        interaction_df = pd.DataFrame(user_item_ids, columns=["user", "item"])
        self.all_interactions = self.all_interactions.append(interaction_df, ignore_index=True)

    def train(self):
        """
        Calculates predicted scores for every user-item pair by first running an
        implicit MF algorithm on the interaction data. If there are no interactions
        yet (such as when the system is first initialized), scores are generated
        using the initialized user & item representations.
        """
        # if there are new interactions present, retrain the
        if self.all_interactions.size > 0:
            # run LensKit ImplicitMF training procedure to extract user/item latent
            # representations from interaction data.
            self.model_params["features"] = self.num_latent_factors
            model = als.ImplicitMF(**self.model_params)
            model.fit(self.all_interactions)
            self.als_model = model
            # update latent representations
            user_index, item_index = list(set(model.user_index_)), list(set(model.item_index_))
            self.users_hat[user_index, :] = model.user_features_
            self.items_hat[:, item_index] = model.item_features_.T
        # update predicted scores
        # when there are no new interactions to refit an MF from, it simply uses the
        # existing representation of user & item attributes to predict scores
        super().train()

    def run(
        self,
        timesteps=50,
        startup=False,
        train_between_steps=False,
        random_items_per_iter=0,
        vary_random_items_per_iter=False,
        repeated_items=True,
        no_new_items=False,
        reset_interactions=True,
    ):
        """
        Just a simple wrapper so that by default, the RS does not refit the ImplicitMF model
        at every timestep of the simulation. See the `run` method in
        :`~models.recommender.BaseRecommender` for more details about what this does.
        """
        if train_between_steps:
            warnings.warn(
                "train_between_steps is set to True. Note that, at each step, this "
                "overwrites the MF model with a model fit only to the latest interaction. "
                "To avoid this behavior, set train_between_steps to False."
            )
        # reset interactions tracker so that interactions captured
        # are only for the duration of this particular run
        if reset_interactions:
            self.all_interactions = pd.DataFrame(columns=["user", "item"])
        super().run(
            timesteps,
            startup,
            train_between_steps,
            random_items_per_iter,
            vary_random_items_per_iter,
            repeated_items,
            no_new_items=no_new_items,
        )

    def startup_and_train(self, timesteps=50, no_new_items=True):
        """
        Wrapper for :`~models.recommender.BaseRecommender` `startup_and_train` method.
        Ensures that by default, no new items are created during the startup training period.
        """
        super().startup_and_train(timesteps, no_new_items=no_new_items)

    def process_new_items(self, new_items):
        """
        Currently, ImplicitMF processes new items by performing a simple mean imputation
        over the latent representations of the existing items. Note: this requires that
        a model has already been fit prior to new items being created.
        """
        num_new_items = new_items.shape[1]
        avg_item = self.als_model.item_features_.T.mean(axis=1)
        new_items = np.tile(avg_item, (num_new_items, 1)).T
        self.items_hat = np.hstack([self.items_hat, new_items])
