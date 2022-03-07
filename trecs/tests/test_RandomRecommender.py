from trecs.metrics.measurement import MSEMeasurement
from trecs.models import RandomRecommender
from trecs.components import Creators
import numpy as np
import scipy.sparse as sp
import pytest
import test_helpers


class TestRandomRecommender:
    def test_default(self):
        r = RandomRecommender()
        test_helpers.assert_correct_num_users(r.num_users, r, r.users_hat.num_users)
        test_helpers.assert_correct_num_items(r.num_items, r, r.items_hat.num_items)
        test_helpers.assert_not_none(r.predicted_scores)

    def test_arguments(self, items=None, users=None):
        if items is None:
            items = np.random.randint(1, 1000)
        if users is None:
            users = np.random.randint(1, 100)

        # init with given arguments
        r = RandomRecommender(num_users=users, num_items=items)
        test_helpers.assert_correct_num_users(users, r, r.users_hat.num_users)
        test_helpers.assert_correct_num_items(items, r, r.items_hat.num_items)
        test_helpers.assert_not_none(r.predicted_scores)

    def test_partial_arguments(self, items=None, users=None):
        # init with partially given arguments
        if items is None:
            items = np.random.randint(1, 1000)
        if users is None:
            users = np.random.randint(1, 100)

        r = RandomRecommender(num_users=users)
        test_helpers.assert_correct_num_users(users, r, r.users_hat.num_users)
        test_helpers.assert_correct_num_items(r.num_items, r, r.items_hat.num_items)
        test_helpers.assert_not_none(r.predicted_scores)

        r = RandomRecommender(num_items=items)
        test_helpers.assert_correct_num_users(r.num_users, r, r.users_hat.num_users)
        test_helpers.assert_correct_num_items(items, r, r.items_hat.num_items)
        test_helpers.assert_not_none(r.predicted_scores)

        r = RandomRecommender(num_users=users, num_items=items)
        test_helpers.assert_correct_num_users(users, r, r.users_hat.num_users)
        test_helpers.assert_correct_num_items(items, r, r.items_hat.num_items)
        test_helpers.assert_not_none(r.predicted_scores)

    def test_representations(self, item_repr=None, user_repr=None):
        if item_repr is None:
            items = np.random.randint(5, 1000)
            item_repr = np.random.random(size=(1, items))
        if user_repr is None or user_repr.shape[1] != item_repr.shape[0]:
            users = np.random.randint(5, 100)
            user_repr = np.random.randint(10, size=(users, 1))

        # init with given arguments
        with pytest.warns(UserWarning):
            # user and item representation should be overwritten
            r = RandomRecommender(item_representation=item_repr, num_items=items)
        test_helpers.assert_correct_num_items(items, r, r.items_hat.num_items)

        with pytest.warns(UserWarning):
            # user and item representation should be overwritten
            r = RandomRecommender(user_representation=user_repr, num_users=users)
        test_helpers.assert_correct_num_users(users, r, r.users_hat.num_users)

    def test_additional_params(self, num_items_per_iter=None):
        if num_items_per_iter is None:
            num_items_per_iter = np.random.randint(5, 100)

        r = RandomRecommender(verbose=False, num_items_per_iter=num_items_per_iter)
        assert num_items_per_iter == r.num_items_per_iter
        # also check other params
        test_helpers.assert_correct_num_users(r.num_users, r, r.users_hat.num_users)
        test_helpers.assert_correct_num_items(r.num_items, r, r.items_hat.num_items)
        test_helpers.assert_not_none(r.predicted_scores)

    def test_seeding(self, seed=None, items=None, users=None):
        if seed is None:
            seed = np.random.randint(100000)
        r1 = RandomRecommender(seed=seed, record_base_state=True)
        r1.add_metrics(MSEMeasurement())
        r2 = RandomRecommender(seed=seed, record_base_state=True)
        r2.add_metrics(MSEMeasurement())
        test_helpers.assert_equal_arrays(r1.items_hat, r2.items_hat)
        test_helpers.assert_equal_arrays(r1.users_hat, r2.users_hat)
        r1.run(timesteps=5)
        r2.run(timesteps=5)
        # check that measurements are the same
        meas1 = r1.get_measurements()
        meas2 = r2.get_measurements()
        test_helpers.assert_equal_measurements(meas1, meas2)
        systate1 = r1.get_system_state()
        systate2 = r2.get_system_state()
        test_helpers.assert_equal_system_state(systate1, systate2)

        if items is None:
            items = np.random.randint(20, 1000)
        if users is None:
            users = np.random.randint(20, 100)
        r1 = RandomRecommender(seed=seed, num_users=users, num_items=items, record_base_state=True)
        r1.add_metrics(MSEMeasurement())
        r2 = RandomRecommender(seed=seed, num_users=users, num_items=items, record_base_state=True)
        r2.add_metrics(MSEMeasurement())
        test_helpers.assert_equal_arrays(r1.items_hat, r2.items_hat)
        test_helpers.assert_equal_arrays(r1.users_hat, r2.users_hat)
        r1.run(timesteps=5)
        r2.run(timesteps=5)
        # check that measurements are the same
        meas1 = r1.get_measurements()
        meas2 = r2.get_measurements()
        test_helpers.assert_equal_measurements(meas1, meas2)
        systate1 = r1.get_system_state()
        systate2 = r2.get_system_state()
        test_helpers.assert_equal_system_state(systate1, systate2)

    def test_recommendations(self):
        num_users = 5
        num_items = 5
        num_attr = 5
        users = np.eye(num_users)  # 5 users, 5 attributes
        items = np.zeros((num_attr, num_items))  # 5 items, 5 attributes
        items[:, 0] = 10  # this item will be most desirable to users

        model = RandomRecommender(
            actual_user_representation=users,
            actual_item_representation=items,
            num_items_per_iter=num_items,
            seed=1234,  # seed ensures that this test passes deterministically
        )

        model.num_items_per_iter = 1
        first_rec = model.recommend()
        init_pred_scores = model.predicted_user_item_scores.copy()
        model.run(1)
        second_rec = model.recommend()
        trained_preds = model.predicted_user_item_scores.copy()

        # assert that item scores have not changed between runs
        test_helpers.assert_equal_arrays(init_pred_scores, trained_preds)
        test_helpers.assert_equal_arrays(init_pred_scores, np.zeros(trained_preds.shape))

        # assert that recommendations are random between runs
        with pytest.raises(AssertionError):
            test_helpers.assert_equal_arrays(first_rec, second_rec)

        # assert that items_hat and users_hat are as expected
        user_rep = np.zeros(num_users).reshape(-1, 1)
        test_helpers.assert_equal_arrays(model.predicted_user_profiles, user_rep)

        item_rep = np.zeros(num_items).reshape(1, -1)
        test_helpers.assert_equal_arrays(model.predicted_item_attributes, item_rep)

    def test_sparse_matrix(self):
        num_users = 5
        num_items = 5
        num_attr = 5
        users = sp.csr_matrix(np.eye(num_users))  # 5 users, 5 attributes
        items = sp.csr_matrix(np.zeros((num_attr, num_items)))  # 5 items, 5 attributes
        items[:, 0] = 10  # this item will be most desirable to users

        model = RandomRecommender(
            actual_user_representation=users.copy(),
            actual_item_representation=items.copy(),
            num_items_per_iter=num_items,
            seed=1234,  # ensures the test passes deterministically
        )
        model.run(1)

        # assert that recommendations are in random order from one iteration to the next
        model.num_items_per_iter = 1
        first_rec = model.recommend()
        model.run(1)
        second_rec = model.recommend()
        with pytest.raises(AssertionError):
            test_helpers.assert_equal_arrays(first_rec, second_rec)

    def test_creator_items(self):
        users = np.random.randint(10, size=(100, 10))
        items = np.random.randint(2, size=(10, 100))
        creator_profiles = Creators(
            np.random.uniform(size=(50, 10)), creation_probability=1.0
        )  # 50 creator profiles
        r = RandomRecommender(
            actual_user_representation=users,
            actual_item_representation=items,
            creators=creator_profiles,
        )
        r.run(1, repeated_items=True)
        assert r.items.num_items == 150  # 50 new items
        assert r.items_hat.num_items == 150
        assert r.users.actual_user_scores.num_items == 150

    def test_new_users(self):
        users = np.random.randint(10, size=(100, 10))
        items = np.random.randint(2, size=(10, 100))
        r = RandomRecommender(
            actual_user_representation=users,
            actual_item_representation=items,
        )
        r.run(1, repeated_items=True)
        num_new_users = 100
        users = np.random.randint(10, size=(num_new_users, 10))
        r.add_users(users)
        # 100 new users + 100 original = 200
        assert r.num_users == 200
        assert r.users.num_users == 200
        assert r.users_hat.num_users == 200
        assert r.users.actual_user_scores.num_users == 200
        r.run(1, repeated_items=True)
