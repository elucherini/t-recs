"""
Implicit MF recommender system
"""
import numpy as np
from lenskit.algorithms import als
import pandas as pd
from trecs.metrics import MSEMeasurement
from trecs.random import Generator
from trecs.validate import validate_user_item_inputs
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
        TODO: fill in

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
        model_params={},
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
            10,
            num_latent_factors,
        )
        self.num_latent_factors = num_attributes
        self.model_params = model_params
        self.all_interactions = pd.DataFrame(columns=["user", "item"])

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

    def update_predicted_scores(self):
        """
        Calculates predicted scores for every user-item pair by first running an
        implicit MF algorithm on the interaction data. A glorified wrapper function,
        but necessary to keep the functionality in BaseRecommender working.
        """
        self.fit_mf()


    def run(self, train_between_steps=False, **kwargs):
        """
        Just a simple wrapper so that by default, the RS does not refit the ImplicitMF model
        at every timestep of the simulation.
        """
        # reset interactions tracker so that interactions captured
        # are only for the duration of this particular run
        self.all_interactions = pd.DataFrame(columns=["user", "item"])
        super().run(
            train_between_steps=train_between_steps,
            **kwargs
        )

    def fit_mf(self):
        """
        Run LensKit ImplicitMF training procedure to extract user/item latent
        representations from interaction data.
        """
        self.model_params["features"] = self.num_latent_factors
        model = als.ImplicitMF(**self.model_params)
        model.fit(self.all_interactions)
        # update latent representations
        self.users_hat, self.items_hat = model.user_features_, model.item_features_.T
        # update predicted scores
        super().update_predicted_scores()

    def process_new_items(self, new_items):
        """
        TODO: fill in
        """
        raise RuntimeError(
            "ImplicitMF has not been designed to work with dynamic items generation yet"
        )
