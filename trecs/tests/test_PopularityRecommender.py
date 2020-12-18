from trecs.models import PopularityRecommender
from trecs.components import Creators
import numpy as np
import pytest
import test_helpers


class TestPopularityRecommender:
    def test_default(self):
        c = PopularityRecommender()
        test_helpers.assert_correct_num_users(c.num_users, c, c.users_hat.shape[0])
        test_helpers.assert_correct_num_items(c.num_items, c, c.items_hat.shape[1])
        test_helpers.assert_not_none(c.predicted_scores)

    def test_arguments(self, items=None, users=None):
        if items is None:
            items = np.random.randint(1, 1000)
        if users is None:
            users = np.random.randint(1, 100)

        # init with given arguments
        c = PopularityRecommender(num_users=users, num_items=items)
        test_helpers.assert_correct_num_users(users, c, c.users_hat.shape[0])
        test_helpers.assert_correct_num_items(items, c, c.items_hat.shape[1])
        test_helpers.assert_not_none(c.predicted_scores)

    def test_partial_arguments(self, items=None, users=None):
        # init with partially given arguments
        if items is None:
            items = np.random.randint(1, 1000)
        if users is None:
            users = np.random.randint(1, 100)

        c = PopularityRecommender(num_users=users)
        test_helpers.assert_correct_num_users(users, c, c.users_hat.shape[0])
        test_helpers.assert_correct_num_items(c.num_items, c, c.items_hat.shape[1])
        test_helpers.assert_not_none(c.predicted_scores)

        c = PopularityRecommender(num_items=items)
        test_helpers.assert_correct_num_users(c.num_users, c, c.users_hat.shape[0])
        test_helpers.assert_correct_num_items(items, c, c.items_hat.shape[1])
        test_helpers.assert_not_none(c.predicted_scores)

        c = PopularityRecommender(num_users=users, num_items=items)
        test_helpers.assert_correct_num_users(users, c, c.users_hat.shape[0])
        test_helpers.assert_correct_num_items(items, c, c.items_hat.shape[1])
        test_helpers.assert_not_none(c.predicted_scores)

    def test_representations(self, item_repr=None, user_repr=None):
        if item_repr is None:
            items = np.random.randint(5, 1000)
            item_repr = np.random.random(size=(1, items))
        if user_repr is None or user_repr.shape[1] != item_repr.shape[0]:
            users = np.random.randint(5, 100)
            user_repr = np.random.randint(10, size=(users, 1))

        c = PopularityRecommender(item_representation=item_repr)
        test_helpers.assert_correct_num_users(c.num_users, c, c.users_hat.shape[0])
        test_helpers.assert_correct_num_items(item_repr.shape[1], c, c.items_hat.shape[1])
        test_helpers.assert_equal_arrays(item_repr, c.items_hat)
        test_helpers.assert_not_none(c.predicted_scores)

        c = PopularityRecommender(user_representation=user_repr)
        test_helpers.assert_correct_num_users(user_repr.shape[0], c, c.users_hat.shape[0])
        test_helpers.assert_correct_num_items(c.num_items, c, c.items_hat.shape[1])
        test_helpers.assert_equal_arrays(user_repr, c.users_hat)
        test_helpers.assert_not_none(c.predicted_scores)

        c = PopularityRecommender(user_representation=user_repr, item_representation=item_repr)
        test_helpers.assert_correct_num_users(user_repr.shape[0], c, c.users_hat.shape[0])
        test_helpers.assert_correct_num_items(item_repr.shape[1], c, c.items_hat.shape[1])
        test_helpers.assert_equal_arrays(user_repr, c.users_hat)
        test_helpers.assert_equal_arrays(item_repr, c.items_hat)
        test_helpers.assert_not_none(c.predicted_scores)

    def test_wrong_representation(
        self, user_repr=None, item_repr=None, bad_user_repr=None, bad_item_repr=None
    ):
        if item_repr is None:
            items = np.random.randint(1000)
            item_repr = np.random.random(size=(1, items))
        if user_repr is None or user_repr.shape[1] != item_repr.shape[0]:
            users = np.random.randint(100)
            user_repr = np.random.randint(10, size=(users, 1))

        if bad_user_repr is None or bad_user_repr.shape[1] == item_repr.shape[0]:
            # |A| shouldn't match item_repr.shape[0]
            bad_user_repr = np.random.randint(10, size=(user_repr.shape[0], user_repr.shape[1] + 2))
        if bad_item_repr is None or bad_item_repr.shape[0] == user_repr.shape[1]:
            # |A| shouldn't match user_repr.shape[1]
            bad_item_repr = np.random.random(size=(item_repr.shape[0] + 1, item_repr.shape[1]))

        with pytest.raises(ValueError):
            c = PopularityRecommender(
                user_representation=bad_user_repr, item_representation=item_repr
            )
        with pytest.raises(ValueError):
            c = PopularityRecommender(
                user_representation=user_repr, item_representation=bad_item_repr
            )

    def test_additional_params(self, num_items_per_iter=None):
        if num_items_per_iter is None:
            num_items_per_iter = np.random.randint(5, 100)

        c = PopularityRecommender(verbose=False, num_items_per_iter=num_items_per_iter)
        assert num_items_per_iter == c.num_items_per_iter
        # also check other params
        test_helpers.assert_correct_num_users(c.num_users, c, c.users_hat.shape[0])
        test_helpers.assert_correct_num_items(c.num_items, c, c.items_hat.shape[1])
        test_helpers.assert_not_none(c.predicted_scores)

    def test_seeding(self, seed=None, items=None, users=None):
        if seed is None:
            seed = np.random.randint(100000)
        s1 = PopularityRecommender(seed=seed, record_base_state=True)
        s2 = PopularityRecommender(seed=seed, record_base_state=True)
        test_helpers.assert_equal_arrays(s1.items_hat, s2.items_hat)
        test_helpers.assert_equal_arrays(s1.users_hat, s2.users_hat)
        s1.run(timesteps=5)
        s2.run(timesteps=5)
        # check that measurements are the same
        meas1 = s1.get_measurements()
        meas2 = s2.get_measurements()
        test_helpers.assert_equal_measurements(meas1, meas2)
        systate1 = s1.get_system_state()
        systate2 = s2.get_system_state()
        test_helpers.assert_equal_system_state(systate1, systate2)

        if items is None:
            items = np.random.randint(20, 1000)
        if users is None:
            users = np.random.randint(20, 100)
        s1 = PopularityRecommender(
            seed=seed, num_users=users, num_items=items, record_base_state=True
        )
        s2 = PopularityRecommender(
            seed=seed, num_users=users, num_items=items, record_base_state=True
        )
        test_helpers.assert_equal_arrays(s1.items_hat, s2.items_hat)
        test_helpers.assert_equal_arrays(s1.users_hat, s2.users_hat)
        s1.run(timesteps=5)
        s2.run(timesteps=5)
        # check that measurements are the same
        meas1 = s1.get_measurements()
        meas2 = s2.get_measurements()
        test_helpers.assert_equal_measurements(meas1, meas2)
        systate1 = s1.get_system_state()
        systate2 = s2.get_system_state()
        test_helpers.assert_equal_system_state(systate1, systate2)

    def test_creator_items(self):
        users = np.random.randint(10, size=(100, 10))
        items = np.random.randint(2, size=(10, 100))
        creator_profiles = Creators(
            np.random.uniform(size=(50, 10)), creation_probability=1.0
        )  # 50 creator profiles
        p = PopularityRecommender(
            actual_user_representation=users,
            actual_item_representation=items,
            creators=creator_profiles,
        )
        p.run(1, repeated_items=True)
        assert p.items.shape[1] == 150  # 50 new items
        assert p.items_hat.shape[1] == 150
        assert p.users.state_history[-1].shape[1] == 150
