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
from .mf import ImplicitMF
from sklearn.metrics import pairwise_distances


class ImplicitMFLFD(ImplicitMF):
    def __init__(
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
        top_n_limit=None,
        **kwargs,
    ):

        super().__init__(
            num_users,
            num_items,
            num_latent_factors,
            user_representation,
            item_representation,
            actual_user_representation,
            actual_item_representation,
            probabilistic_recommendations,
            seed,
            verbose,
            num_items_per_iter,
            model_params,
            **kwargs,
        )

        if not top_n_limit:
            self.top_n_limit = self.items_hat.shape[1]
        else:
            self.top_n_limit=top_n_limit

    def generate_recommendations(self, k=1, item_indices=None):
        """
        Generate recommendations for each user based on latent feature diversification algorithm in Willemsen et al., (2016).
        Method works by taking the highest predicted item, then iteratively adding items that are maximally distant from the centroid of the attributes of items
        already in the recommendation list

        Parameters
        -----------


            k : int (optional, default: 1)
                Number of items to recommend.
            top_n_limit: int (optional, default: None)
                Constraint on search for items to the top n
            item_indices : :obj:`numpy.ndarray` or None (optional, default: None)
                A matrix containing the indices of the items each user has not yet
                interacted with. It is used to ensure that the user is presented
                with items they have not already interacted with. If `None`,
                then the user may be recommended items that they have already
                interacted with.

        Returns
        ---------
            Recommendations: :obj:`numpy.ndarray`
        """

        if item_indices is not None:
            if item_indices.size < self.num_users:
                raise ValueError(
                    "At least one user has interacted with all items!"
                    "To avoid this problem, you may want to allow repeated items."
                )
            if k > item_indices.shape[1]:
                raise ValueError(
                    f"There are not enough items left to recommend {k} items to each user."
                )
            if k == 0:
                return np.array([]).reshape((self.num_users, 0)).astype(int)



        row = np.repeat(self.users.user_vector, item_indices.shape[1])
        row = row.reshape((self.num_users, -1))
        s_filtered = self.predicted_scores[row, item_indices]

        negated_scores = -1 * s_filtered  # negate scores so indices go from highest to lowest
        # break ties using a random score component
        scores_tiebreak = np.zeros(negated_scores.shape, dtype=[("score", "f8"), ("random", "f8")])
        scores_tiebreak["score"] = negated_scores
        scores_tiebreak["random"] = self.random_state.random(negated_scores.shape)
        top_k = scores_tiebreak.argpartition(self.top_n_limit - 1, order=["score", "random"])[
            :, :self.top_n_limit
        ]
        # now we sort within the top k
        row = np.repeat(self.users.user_vector, self.top_n_limit).reshape((self.num_users, -1))
        # again, indices should go from highest to lowest
        sort_top_k = scores_tiebreak[row, top_k].argsort(order=["score", "random"])
        top_k_recs = item_indices[row, top_k[row, sort_top_k]]

        # dims are attribute, items, users
        top_k_att = self.items_hat[:, top_k_recs[:]].swapaxes(1, 2)

        rec = []
        for idx, user in enumerate(self.users_hat):

            # make a copy so as not to modify the original array
            user_item_feats = np.array(top_k_att[:, :, idx])

            orig_user_item_feats = np.array(user_item_feats)
            # user_item_feats_idx = [0]
            user_max_idx = top_k_recs[idx, 0]
            recs_idxs = [user_max_idx]

            # hold the features of the recommended items
            recs_features = self.items_hat[:, user_max_idx]

            for r in range(1, k):

                if r == 1:
                    # for the second item, just use the first item values
                    centroid = recs_features
                else:
                    centroid = np.nanmean(recs_features, axis=0)

                centroid = centroid.reshape(1, -1)

                # set all the previously chosen item features to the centroid, so they will not be selected again
                # don't want to just remove rows because it will throw off the indexing
                user_item_feats[:, 0 : r + 1] = centroid.T

                d = pairwise_distances(
                    X=centroid,
                    Y=user_item_feats.T,
                    metric="cityblock",
                    force_all_finite="allow_nan",
                )

                most_distant = np.argmax(d)

                # distances.append(d.max())

                # most_distant_feats = user_item_feats.T[most_distant]

                # get the index of the most distant item in the top k recs
                recs_idxs.append(top_k_recs[idx, most_distant])
                recs_features = np.vstack((recs_features, user_item_feats[:, most_distant]))

            rec.append(recs_idxs)
        self.rec = rec

        return np.array(rec)
