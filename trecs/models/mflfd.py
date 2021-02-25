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
            **kwargs
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

    def generate_recommendations(self, k=1, item_indices=None):
        """
        Generate recommendations for each user.

        Parameters
        -----------

            k : int (optional, default: 1)
                Number of items to recommend.

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
        self.item_indices = item_indices
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
        if self.probabilistic_recommendations:
            permutation = s_filtered.argsort()
            rec = item_indices[row, permutation]
            # the recommended items will not be exactly determined by
            # predicted score; instead, we will sample from the sorted list
            # such that higher-preference items get more probability mass
            num_items_unseen = rec.shape[1]  # number of items unseen per user
            probabilities = np.logspace(0.0, num_items_unseen / 10.0, num=num_items_unseen, base=2)
            probabilities = probabilities / probabilities.sum()
            picks = np.random.choice(num_items_unseen, k, replace=False, p=probabilities)
            return rec[:, picks]
        else:
            # scores are U x I; we can use argpartition to take the top k scores
            negated_scores = -1 * s_filtered  # negate scores so indices go from highest to lowest
            # break ties using a random score component
            scores_tiebreak = np.zeros(
                negated_scores.shape, dtype=[("score", "f8"), ("random", "f8")]
            )
            scores_tiebreak["score"] = negated_scores
            scores_tiebreak["random"] = self.random_state.random(negated_scores.shape)
            top_k = scores_tiebreak.argpartition(k - 1, order=["score", "random"])[:, :k]
            # now we sort within the top k
            row = np.repeat(self.users.user_vector, k).reshape((self.num_users, -1))
            # again, indices should go from highest to lowest
            sort_top_k = scores_tiebreak[row, top_k].argsort(order=["score", "random"])
            rec = item_indices[
                row, top_k[row, sort_top_k]
            ]  # extract items such that rows go from highest scored to lowest-scored of top-k
            if self.is_verbose():
                self.log(f"Item indices:\n{str(item_indices)}")
                self.log(
                    f"Top-k items ordered by preference (high to low) for each user:\n{str(rec)}"
                )
            self.rec = rec
            return rec

    # def latent_factors_diversification(self, top_n_limit=None):
    #
    #         #(user_features, item_features, n_recs=10, top_n_limit=None):
    #
    #     hat_ratings = np.dot(user_features, item_features.T)
    #
    #     if top_n_limit:
    #         # if constraining by top n, only retain the top n ratings within each user
    #         ind = np.argpartition(hat_ratings, -top_n_limit)[:, -top_n_limit:]
    #         n_ratings = np.take(hat_ratings, ind)
    #     else:
    #         # if not constraining by top n, retail all item indices for all users.
    #         # If this is the case, in all_user_recs, recs_idxs should match original_recs_idxs
    #         ind = np.tile(np.arange(0, len(item_features)), (len(user_features), 1))
    #         n_ratings = hat_ratings
    #
    #     all_user_recs = dict()
    #
    #     max_idx = np.argmax(n_ratings, axis=1)
    #     top_items = item_features[max_idx]
    #
    #     all_recs = np.empty([user_features.shape[0], item_features.shape[1], n_recs])
    #     # all_recs = None
    #
    #     for idx, user in enumerate(user_features):
    #
    #         user_item_feats = item_features[ind[idx]]
    #         user_max_idx = np.argmax(n_ratings[idx])
    #
    #         # get the top rec and add that as the first item for each user
    #         user_max = max_idx[idx]
    #         recs_features = top_items[idx]
    #         recs_idxs = [max_idx[idx]]
    #         recs_preds = [n_ratings[idx][user_max]]
    #         orig_recs_idxs = [ind[idx, user_max]]
    #
    #         for rec in range(1, n_recs):
    #             if rec == 1:
    #                 # for the second item, just use the first item values
    #                 centroid = recs_features
    #             else:
    #                 centroid = np.nanmean(recs_features, axis=0)
    #
    #             centroid = centroid.reshape(1, -1)
    #
    #             # set all the previously chosen item features to the centroid, so they will not be selected again
    #             # don't want to just remove rows because it will throw of the indexing
    #             user_item_feats[recs_idxs] = centroid
    #
    #             d = pairwise_distances(X=centroid, Y=user_item_feats, metric='cityblock', force_all_finite='allow_nan')
    #             most_distant = np.argmax(d)
    #
    #             recs_idxs.append(most_distant)
    #             # get the item index from the original array of indices, not the constrained array
    #             orig_recs_idxs.append(ind[idx, most_distant])
    #             recs_preds.append(n_ratings[idx][most_distant])
    #
    #             recs_features = np.vstack((recs_features, user_item_feats[most_distant]))
    #
    #         all_recs[idx, :, :] = recs_features
    #
    #         all_user_recs[idx] = {'user_feats': user,
    #                               'original_recs_idx': orig_recs_idxs,
    #                               'recs_idx': recs_idxs,
    #                               'recs_features': recs_features,
    #                               'recs_preds': recs_preds}
    #
    #     return all_recs, all_user_recs

    # def generate_recommendations(self, k=1, item_indices=None):
    #
    #     if item_indices is not None:
    #         if item_indices.size < self.num_users:
    #             raise ValueError(
    #                 "At least one user has interacted with all items!"
    #                 "To avoid this problem, you may want to allow repeated items."
    #             )
    #         if k > item_indices.shape[1]:
    #             raise ValueError(
    #                 f"There are not enough items left to recommend {k} items to each user."
    #             )
    #     if k == 0:
    #         return np.array([]).reshape((self.num_users, 0)).astype(int)
    #     row = np.repeat(self.users.user_vector, item_indices.shape[1])
    #     row = row.reshape((self.num_users, -1))
    #     s_filtered = self.predicted_scores[row, item_indices]
    #
    #     if self.probabilistic_recommendations:
    #         permutation = s_filtered.argsort()
    #         rec = item_indices[row, permutation]
    #         # the recommended items will not be exactly determined by
    #         # predicted score; instead, we will sample from the sorted list
    #         # such that higher-preference items get more probability mass
    #         num_items_unseen = rec.shape[1]  # number of items unseen per user
    #         probabilities = np.logspace(0.0, num_items_unseen / 10.0, num=num_items_unseen, base=2)
    #         probabilities = probabilities / probabilities.sum()
    #         picks = np.random.choice(num_items_unseen, k, replace=False, p=probabilities)
    #         return rec[:, picks]
    #     else:
    #         # scores are U x I; we can use argpartition to take the top k scores
    #         negated_scores = -1 * s_filtered  # negate scores so indices go from highest to lowest
    #         # break ties using a random score component
    #         scores_tiebreak = np.zeros(
    #             negated_scores.shape, dtype=[("score", "f8"), ("random", "f8")]
    #         )
    #         scores_tiebreak["score"] = negated_scores
    #         scores_tiebreak["random"] = self.random_state.random(negated_scores.shape)
    #         top_k = scores_tiebreak.argpartition(k - 1, order=["score", "random"])[:, :k]
    #         # now we sort within the top k
    #         row = np.repeat(self.users.user_vector, k).reshape((self.num_users, -1))
    #         # again, indices should go from highest to lowest
    #         sort_top_k = scores_tiebreak[row, top_k].argsort(order=["score", "random"])
    #
    #
    #         rec = item_indices[
    #             row, top_k[row, sort_top_k]
    #         ]  # extract items such that rows go from highest scored to lowest-scored of top-k
    #         if self.is_verbose():
    #             self.log(f"Item indices:\n{str(item_indices)}")
    #             self.log(
    #                 f"Top-k items ordered by preference (high to low) for each user:\n{str(rec)}"
    #             )
    #         return rec