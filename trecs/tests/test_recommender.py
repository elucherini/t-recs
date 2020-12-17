import numpy as np
from trecs.models import BaseRecommender
from trecs.components import Creators
import test_helpers


class DummyRecommender(BaseRecommender):
    def __init__(
        self, users_hat, items_hat, users, items, num_users, num_items, num_items_per_iter, **kwargs
    ):
        super().__init__(
            users_hat, items_hat, users, items, num_users, num_items, num_items_per_iter, **kwargs
        )

    def _update_user_profiles(self, interactions):
        pass

    def process_new_items(self, new_items):
        # generate a representation of ones
        num_attr = self.items.shape[0]
        num_items = new_items.shape[1]
        self.items_hat = np.hstack([self.items_hat, np.random.uniform(size=(num_attr, num_items))])


class TestBaseRecommender:
    # 10 users and 50 items
    users = np.random.randint(10, size=(10, 5))
    items = np.random.randint(10, size=(5, 50))
    users_hat = np.copy(users)
    items_hat = np.copy(items)

    def test_generate_recommendations(self):
        dummy = DummyRecommender(self.users_hat, self.items_hat, self.users, self.items, 10, 50, 5)
        # recommend 5 items at this timestep
        recs = dummy.generate_recommendations(k=5, item_indices=dummy.indices)
        # assert that the recommendations have dimensions
        # (number of users) x (number of items to recommend per user)
        assert recs.shape[1] == 5
        assert recs.shape[0] == 10

    def test_interaction_indices(self):
        # show 5 items per iteration
        dummy = DummyRecommender(self.users_hat, self.items_hat, self.users, self.items, 10, 50, 5)
        for i in range(10):
            dummy.run(1, repeated_items=False)  # run 1 timestep
            # check that the number of interactions is divisible by the
            # number of users
            assert (dummy.indices == -1).sum() % dummy.num_users == 0
            # each user interacts with one new item
            assert (dummy.indices == -1).sum() == (i + 1) * 10

    def test_repeated_items(self):
        # show 5 items per iteration
        dummy = DummyRecommender(self.users_hat, self.items_hat, self.users, self.items, 10, 50, 5)
        dummy.run(5, repeated_items=True)  # run 5 timesteps
        # check that no users are recorded as having interacted with items
        assert (dummy.indices == -1).sum() == 0

    def test_closed_logger(self):
        dummy = DummyRecommender(self.users_hat, self.items_hat, self.users, self.items, 10, 50, 5)
        dummy.run(5, repeated_items=True)  # run 5 timesteps
        logger = dummy._logger.logger  # pylint: disable=protected-access
        handler = dummy._logger.handler  # pylint: disable=protected-access
        assert len(logger.handlers) > 0  # before garbage collection
        del dummy
        # after garbage collection, handler should be closed
        assert handler not in logger.handlers

    def test_content_creators(self):
        # 10 content creators
        creators = Creators(np.random.uniform(size=(10, 5)), creation_probability=1)
        dummy = DummyRecommender(
            self.users_hat, self.items_hat, self.users, self.items, 10, 50, 5, creators=creators
        )
        assert dummy.num_items == 50
        dummy.run(5, repeated_items=True)  # run 5 timesteps
        assert dummy.num_items == 100  # 10 creators * 5 iterations + 50 initial items
        # assert scores are updated correctly
        created_items = dummy.items_hat[:, 50:100]
        true_scores = self.users @ created_items
        predicted_scores = dummy.predicted_scores[:, 50:100]
        # the predicted scores normalize the user arrays before doing the dot product,
        # so instead we verify the sorted position of each item
        test_helpers.assert_equal_arrays(true_scores.argsort(), predicted_scores.argsort())
