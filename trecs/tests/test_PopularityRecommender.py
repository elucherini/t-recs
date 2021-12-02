from trecs.metrics.measurement import MSEMeasurement
from trecs.models import PopularityRecommender
from trecs.components import Creators
import numpy as np
import scipy.sparse as sp
import pytest
import test_helpers


class TestPopularityRecommender:
    def test_default(self):
        p = PopularityRecommender()
        test_helpers.assert_correct_num_users(p.num_users, p, p.users_hat.num_users)
        test_helpers.assert_correct_num_items(p.num_items, p, p.items_hat.num_items)
        test_helpers.assert_not_none(p.predicted_scores)

    def test_arguments(self, items=None, users=None):
        if items is None:
            items = np.random.randint(1, 1000)
        if users is None:
            users = np.random.randint(1, 100)

        # init with given arguments
        p = PopularityRecommender(num_users=users, num_items=items)
        test_helpers.assert_correct_num_users(users, p, p.users_hat.num_users)
        test_helpers.assert_correct_num_items(items, p, p.items_hat.num_items)
        test_helpers.assert_not_none(p.predicted_scores)

    def test_partial_arguments(self, items=None, users=None):
        # init with partially given arguments
        if items is None:
            items = np.random.randint(1, 1000)
        if users is None:
            users = np.random.randint(1, 100)

        p = PopularityRecommender(num_users=users)
        test_helpers.assert_correct_num_users(users, p, p.users_hat.num_users)
        test_helpers.assert_correct_num_items(p.num_items, p, p.items_hat.num_items)
        test_helpers.assert_not_none(p.predicted_scores)

        p = PopularityRecommender(num_items=items)
        test_helpers.assert_correct_num_users(p.num_users, p, p.users_hat.num_users)
        test_helpers.assert_correct_num_items(items, p, p.items_hat.num_items)
        test_helpers.assert_not_none(p.predicted_scores)

        p = PopularityRecommender(num_users=users, num_items=items)
        test_helpers.assert_correct_num_users(users, p, p.users_hat.num_users)
        test_helpers.assert_correct_num_items(items, p, p.items_hat.num_items)
        test_helpers.assert_not_none(p.predicted_scores)

    def test_representations(self, item_repr=None, user_repr=None):
        if item_repr is None:
            items = np.random.randint(5, 1000)
            item_repr = np.random.random(size=(1, items))
        if user_repr is None or user_repr.shape[1] != item_repr.shape[0]:
            users = np.random.randint(5, 100)
            user_repr = np.random.randint(10, size=(users, 1))

        p = PopularityRecommender(item_representation=item_repr)
        test_helpers.assert_correct_num_users(p.num_users, p, p.users_hat.num_users)
        test_helpers.assert_correct_num_items(item_repr.shape[1], p, p.items_hat.num_items)
        test_helpers.assert_equal_arrays(item_repr, p.items_hat)
        test_helpers.assert_not_none(p.predicted_scores)

        p = PopularityRecommender(user_representation=user_repr)
        test_helpers.assert_correct_num_users(user_repr.shape[0], p, p.users_hat.num_users)
        test_helpers.assert_correct_num_items(p.num_items, p, p.items_hat.num_items)
        test_helpers.assert_equal_arrays(user_repr, p.users_hat)
        test_helpers.assert_not_none(p.predicted_scores)

        p = PopularityRecommender(user_representation=user_repr, item_representation=item_repr)
        test_helpers.assert_correct_num_users(user_repr.shape[0], p, p.users_hat.num_users)
        test_helpers.assert_correct_num_items(item_repr.shape[1], p, p.items_hat.num_items)
        test_helpers.assert_equal_arrays(user_repr, p.users_hat)
        test_helpers.assert_equal_arrays(item_repr, p.items_hat)
        test_helpers.assert_not_none(p.predicted_scores)

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
            p = PopularityRecommender(
                user_representation=bad_user_repr, item_representation=item_repr
            )
        with pytest.raises(ValueError):
            p = PopularityRecommender(
                user_representation=user_repr, item_representation=bad_item_repr
            )

    def test_additional_params(self, num_items_per_iter=None):
        if num_items_per_iter is None:
            num_items_per_iter = np.random.randint(5, 100)

        p = PopularityRecommender(verbose=False, num_items_per_iter=num_items_per_iter)
        assert num_items_per_iter == p.num_items_per_iter
        # also check other params
        test_helpers.assert_correct_num_users(p.num_users, p, p.users_hat.num_users)
        test_helpers.assert_correct_num_items(p.num_items, p, p.items_hat.num_items)
        test_helpers.assert_not_none(p.predicted_scores)

    def test_seeding(self, seed=None, items=None, users=None):
        if seed is None:
            seed = np.random.randint(100000)
        s1 = PopularityRecommender(seed=seed, record_base_state=True)
        s1.add_metrics(MSEMeasurement())
        s2 = PopularityRecommender(seed=seed, record_base_state=True)
        s2.add_metrics(MSEMeasurement())
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
        s1.add_metrics(MSEMeasurement())
        s2 = PopularityRecommender(
            seed=seed, num_users=users, num_items=items, record_base_state=True
        )
        s2.add_metrics(MSEMeasurement())
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

    def test_recommendations(self):
        num_users = 5
        num_items = 5
        num_attr = 5
        users = np.eye(num_users)  # 5 users, 5 attributes
        items = np.zeros((num_attr, num_items))  # 5 items, 5 attributes
        items[:, 0] = 10  # this item will be most desirable to users

        model = PopularityRecommender(
            actual_user_representation=users,
            actual_item_representation=items,
            num_items_per_iter=num_items,
        )
        init_pred_scores = model.predicted_user_item_scores.copy()
        # after one iteration of training, the most popular item
        # should be the item at index 0
        model.run(1)

        # assert new scores have changed
        trained_preds = model.predicted_user_item_scores.copy()
        with pytest.raises(AssertionError):
            test_helpers.assert_equal_arrays(init_pred_scores, trained_preds)

        # assert that recommendations are now "perfect"
        model.num_items_per_iter = 1
        recommendations = model.recommend()
        correct_rec = np.array([[0], [0], [0], [0], [0]])
        test_helpers.assert_equal_arrays(recommendations, correct_rec)

        # assert that items_hat and users_hat are as expected
        user_rep = np.ones(num_users).reshape(-1, 1)
        test_helpers.assert_equal_arrays(model.predicted_user_profiles, user_rep)

        item_rep = np.zeros(num_items).reshape(1, -1)
        item_rep[0, 0] = 5  # all users should have interacted with this item
        test_helpers.assert_equal_arrays(model.predicted_item_attributes, item_rep)

        # new model that only shows 2 items per iteration
        model = PopularityRecommender(
            actual_user_representation=users, actual_item_representation=items, num_items_per_iter=2
        )
        model.run(5)  # run for 5 timesteps

        # assert that recommendations are now "perfect"
        model.num_items_per_iter = 1
        recommendations = model.recommend()
        most_popular = np.argmax(model.predicted_item_attributes)  # extract most popular item
        correct_rec = np.ones(num_users).reshape(-1, 1) * most_popular
        test_helpers.assert_equal_arrays(recommendations, correct_rec)

    def test_sparse_matrix(self):
        num_users = 5
        num_items = 5
        num_attr = 5
        users = sp.csr_matrix(np.eye(num_users))  # 5 users, 5 attributes
        items = sp.csr_matrix(np.zeros((num_attr, num_items)))  # 5 items, 5 attributes
        users_hat = sp.csr_matrix(np.ones((num_users, 1)))
        items_hat = sp.csr_matrix((1, num_items))
        items[:, 0] = 10  # this item will be most desirable to users

        model = PopularityRecommender(
            user_representation=users_hat.copy(),
            item_representation=items_hat.copy(),
            actual_user_representation=users.copy(),
            actual_item_representation=items.copy(),
            num_items_per_iter=num_items,
        )
        model.run(1)

        # assert that recommendations are now "perfect"
        model.num_items_per_iter = 1
        recommendations = model.recommend()
        correct_rec = np.array([[0], [0], [0], [0], [0]])
        test_helpers.assert_equal_arrays(recommendations, correct_rec)

        # test various combinations of sparse matrix arguments,
        # ensure they run without error
        model = PopularityRecommender(
            user_representation=users_hat.copy(),
            item_representation=items_hat.copy(),
            num_items_per_iter=num_items,
        )
        model.run(1)

        model = PopularityRecommender(
            actual_user_representation=users.copy(),
            actual_item_representation=items.copy(),
            num_items_per_iter=num_items,
        )
        model.run(1)

        model = PopularityRecommender(
            user_representation=users_hat.copy(),
            num_items=num_items,
            num_items_per_iter=num_items,
        )
        model.run(1)

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
        assert p.items.num_items == 150  # 50 new items
        assert p.items_hat.num_items == 150
        assert p.users.actual_user_scores.num_items == 150

    def test_new_users(self):
        users = np.random.randint(10, size=(100, 10))
        items = np.random.randint(2, size=(10, 100))
        p = PopularityRecommender(
            actual_user_representation=users,
            actual_item_representation=items,
        )
        p.run(1, repeated_items=True)
        num_new_users = 100
        users = np.random.randint(10, size=(num_new_users, 10))
        p.add_users(users)
        # 100 new users + 100 original = 200
        assert p.num_users == 200
        assert p.users.num_users == 200
        assert p.users_hat.num_users == 200
        assert p.users.actual_user_scores.num_users == 200
        p.run(1, repeated_items=True)
        # verify the user representation
        user_representation = np.ones(200).reshape(-1, 1)
        test_helpers.assert_equal_arrays(user_representation, p.users_hat)
        # the first iteration should have yielded
        # 100 interactions, the second should yield another 200
        assert p.items_hat.value.sum() == 300
