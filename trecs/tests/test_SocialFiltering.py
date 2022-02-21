import test_helpers
import pytest
import numpy as np
import scipy.sparse as sp
from trecs.metrics.measurement import MSEMeasurement
import trecs.matrix_ops as mo
from trecs.models import SocialFiltering, social
from trecs.components import Creators


class TestSocialFiltering:
    def test_default(self):
        s = SocialFiltering()
        test_helpers.assert_correct_num_users(s.num_users, s, s.users_hat.num_users)
        test_helpers.assert_correct_num_users(s.num_users, s, s.users_hat.num_attrs)
        test_helpers.assert_correct_num_items(s.num_items, s, s.items_hat.num_items)
        test_helpers.assert_not_none(s.predicted_scores)
        # did not set seed, show random behavior
        s1 = SocialFiltering()

        with pytest.raises(AssertionError):
            test_helpers.assert_equal_arrays(s.users_hat, s1.users_hat)

    def test_arguments(self, items=10, users=5):
        if items is None:
            items = np.random.randint(10, 1000)
        if users is None:
            users = np.random.randint(10, 100)
        s = SocialFiltering(num_users=users, num_items=items, seed=1234)
        test_helpers.assert_correct_num_users(users, s, s.users_hat.num_users)
        test_helpers.assert_correct_num_users(users, s, s.users_hat.num_attrs)
        test_helpers.assert_correct_num_items(items, s, s.items_hat.num_items)
        test_helpers.assert_not_none(s.predicted_scores)
        # set different seed, show random behavior
        s1 = SocialFiltering(num_users=users, num_items=items, seed=4321)

        with pytest.raises(AssertionError):
            test_helpers.assert_equal_arrays(s.users_hat, s1.users_hat)

    def test_partial_arguments(self, items=10, users=5):
        if items is None:
            items = np.random.randint(10, 1000)
        if users is None:
            users = np.random.randint(10, 100)
        # init with partially given arguments
        s = SocialFiltering(num_users=users)
        test_helpers.assert_correct_num_users(users, s, s.users_hat.num_users)
        test_helpers.assert_correct_num_users(users, s, s.users_hat.num_attrs)
        test_helpers.assert_correct_num_items(s.num_items, s, s.items_hat.num_items)
        test_helpers.assert_not_none(s.predicted_scores)

        # did not set seed, show random behavior
        s1 = SocialFiltering(num_users=users)

        with pytest.raises(AssertionError):
            test_helpers.assert_equal_arrays(s.users_hat, s1.users_hat)

        s = SocialFiltering(num_items=items)
        test_helpers.assert_correct_num_users(s.num_users, s, s.users_hat.num_users)
        test_helpers.assert_correct_num_users(s.num_users, s, s.users_hat.num_attrs)
        test_helpers.assert_correct_num_items(items, s, s.items_hat.num_items)
        test_helpers.assert_not_none(s.predicted_scores)

        # did not set seed, show random behavior
        s1 = SocialFiltering(num_items=items)

        with pytest.raises(AssertionError):
            test_helpers.assert_equal_arrays(s.users_hat, s1.users_hat)

    def test_representations(self, item_repr=None, user_repr=None):
        if item_repr is None:
            items = np.random.randint(20, 1000)
            users = user_repr.shape[0] if user_repr is not None else np.random.randint(20, 100)
            item_repr = np.random.random(size=(users, items))
        if user_repr is None or user_repr.shape[0] != user_repr.shape[1]:
            users = item_repr.shape[0]
            user_repr = np.random.randint(2, size=(users, users))
        # test item representation
        s = SocialFiltering(item_representation=item_repr)
        test_helpers.assert_correct_num_users(s.num_users, s, s.users_hat.num_users)
        test_helpers.assert_correct_num_users(s.num_users, s, s.users_hat.num_attrs)
        test_helpers.assert_correct_num_items(item_repr.shape[1], s, s.items_hat.num_items)
        test_helpers.assert_equal_arrays(item_repr, s.items_hat)
        test_helpers.assert_not_none(s.predicted_scores)

        # did not set seed, show random behavior
        s1 = SocialFiltering(item_representation=item_repr)

        with pytest.raises(AssertionError):
            test_helpers.assert_equal_arrays(s.users_hat, s1.users_hat)

        # test user representation
        s = SocialFiltering(user_representation=user_repr)
        test_helpers.assert_correct_num_users(user_repr.shape[0], s, s.users_hat.num_users)
        test_helpers.assert_correct_num_users(user_repr.shape[0], s, s.users_hat.num_attrs)
        test_helpers.assert_correct_num_users(user_repr.shape[1], s, s.users_hat.num_users)
        test_helpers.assert_correct_num_users(user_repr.shape[1], s, s.users_hat.num_attrs)
        test_helpers.assert_correct_num_items(s.num_items, s, s.items_hat.num_items)
        test_helpers.assert_equal_arrays(user_repr, s.users_hat)
        test_helpers.assert_not_none(s.predicted_scores)

        # test item and user representations
        s = SocialFiltering(user_representation=user_repr, item_representation=item_repr)
        test_helpers.assert_correct_num_users(user_repr.shape[0], s, s.users_hat.num_users)
        test_helpers.assert_correct_num_users(user_repr.shape[0], s, s.users_hat.num_attrs)
        test_helpers.assert_correct_num_users(user_repr.shape[1], s, s.users_hat.num_users)
        test_helpers.assert_correct_num_users(user_repr.shape[1], s, s.users_hat.num_attrs)
        test_helpers.assert_correct_num_items(item_repr.shape[1], s, s.items_hat.num_items)
        test_helpers.assert_equal_arrays(user_repr, s.users_hat)
        test_helpers.assert_equal_arrays(item_repr, s.items_hat)
        test_helpers.assert_not_none(s.predicted_scores)

    def test_wrong_representations(self, bad_user_repr=None):
        if bad_user_repr is None or bad_user_repr.shape[0] == bad_user_repr.shape[1]:
            # bad_user_repr should not be a square matrix
            users = np.random.randint(10, 100)
            bad_user_repr = np.random.randint(2, size=(users + 2, users))
        # this should throw an exception
        with pytest.raises(ValueError):
            s = SocialFiltering(user_representation=bad_user_repr)

    def test_additional_params(self, num_items_per_iter=None):
        if num_items_per_iter is None:
            num_items_per_iter = np.random.randint(5, 100)
        s = SocialFiltering(verbose=False, num_items_per_iter=num_items_per_iter)
        assert num_items_per_iter == s.num_items_per_iter
        # also check other params
        test_helpers.assert_correct_num_users(s.num_users, s, s.users_hat.num_users)
        test_helpers.assert_correct_num_users(s.num_users, s, s.users_hat.num_attrs)
        test_helpers.assert_correct_num_items(s.num_items, s, s.items_hat.num_items)
        test_helpers.assert_not_none(s.predicted_scores)

        # did not set seed, show random behavior
        s1 = SocialFiltering(verbose=False, num_items_per_iter=num_items_per_iter)

        with pytest.raises(AssertionError):
            test_helpers.assert_equal_arrays(s.users_hat, s1.users_hat)

    def test_social_graph(self, user_repr=None, user1=None, user2=None):
        if user_repr is None or user_repr.shape[0] != user_repr.shape[1]:
            users = np.random.randint(20, 100)
            user_repr = np.zeros((users, users))
        s = SocialFiltering(user_representation=user_repr)
        if user1 is None:
            user1 = np.random.randint(1, s.num_users)
        if user2 is None:
            user2 = np.random.randint(1, s.num_users)
        # users must be different
        while user1 == user2:
            user1 = np.random.randint(1, s.num_users)
        # test current graph
        test_helpers.assert_social_graph_not_following(s.users_hat.value, user1, user2)
        test_helpers.assert_social_graph_not_following(s.users_hat.value, user2, user1)
        # test follow
        s.follow(user1, user2)
        test_helpers.assert_social_graph_following(s.users_hat.value, user1, user2)
        test_helpers.assert_social_graph_not_following(s.users_hat.value, user2, user1)
        # test follow again -- nothing should change
        s.follow(user1, user2)
        test_helpers.assert_social_graph_following(s.users_hat.value, user1, user2)
        test_helpers.assert_social_graph_not_following(s.users_hat.value, user2, user1)
        # test unfollow
        s.unfollow(user1, user2)
        test_helpers.assert_social_graph_not_following(s.users_hat.value, user1, user2)
        test_helpers.assert_social_graph_not_following(s.users_hat.value, user2, user1)
        # test unfollow again -- nothing should change
        s.unfollow(user1, user2)
        test_helpers.assert_social_graph_not_following(s.users_hat.value, user1, user2)
        test_helpers.assert_social_graph_not_following(s.users_hat.value, user2, user1)

        # test friending
        s.add_friends(user1, user2)
        test_helpers.assert_social_graph_following(s.users_hat.value, user1, user2)
        test_helpers.assert_social_graph_following(s.users_hat.value, user2, user1)
        # test friending again -- nothing should change
        s.add_friends(user2, user1)
        test_helpers.assert_social_graph_following(s.users_hat.value, user1, user2)
        test_helpers.assert_social_graph_following(s.users_hat.value, user2, user1)
        # test unfriending
        s.remove_friends(user1, user2)
        test_helpers.assert_social_graph_not_following(s.users_hat.value, user1, user2)
        test_helpers.assert_social_graph_not_following(s.users_hat.value, user2, user1)
        # test unfriending again -- nothing should change
        s.remove_friends(user1, user2)
        test_helpers.assert_social_graph_not_following(s.users_hat.value, user1, user2)
        test_helpers.assert_social_graph_not_following(s.users_hat.value, user2, user1)

    def test_seeding(self, seed=None, items=None, users=None):
        if seed is None:
            seed = np.random.randint(100000)
        s1 = SocialFiltering(seed=seed, record_base_state=True)
        s1.add_metrics(MSEMeasurement())
        s2 = SocialFiltering(seed=seed, record_base_state=True)
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
            users = np.random.randint(1, 100)
        s1 = SocialFiltering(seed=seed, num_users=users, num_items=items, record_base_state=True)
        s1.add_metrics(MSEMeasurement())
        s2 = SocialFiltering(seed=seed, num_users=users, num_items=items, record_base_state=True)
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
        users = np.eye(num_users)  # 5 users, 5 attributes
        items = np.eye(num_items)  # 5 users, 5 attributes
        social_network = np.roll(users, 1, axis=1)  # every user i is connected to (i+1) % 5

        model = SocialFiltering(
            user_representation=social_network,
            actual_item_representation=items,
            actual_user_representation=users,
            num_items_per_iter=num_items,
        )
        init_pred_scores = model.predicted_user_item_scores.copy()
        model.run(1)

        # assert new scores have changed
        trained_preds = model.predicted_user_item_scores.copy()
        with pytest.raises(AssertionError):
            test_helpers.assert_equal_arrays(init_pred_scores, trained_preds)

        # assert that recommendations are now "perfect"
        # every user should be recommended the item that
        # the person they are "following" interacted with
        model.num_items_per_iter = 1
        recommendations = model.recommend()
        correct_rec = np.array([[1], [2], [3], [4], [0]])
        test_helpers.assert_equal_arrays(recommendations, correct_rec)

    def test_sparse_matrix(self):
        num_users = 5
        num_items = 5
        users = sp.csr_matrix(np.eye(num_users))  # 5 users, 5 attributes
        items = sp.csr_matrix(np.eye(num_items))  # 5 users, 5 attributes
        social_network = sp.csr_matrix(
            np.roll(np.eye(num_users), 1, axis=1)
        )  # every user i is connected to (i+1) % 5

        model = SocialFiltering(
            user_representation=social_network.copy(),
            actual_item_representation=items.copy(),
            actual_user_representation=users.copy(),
            num_items_per_iter=num_items,
        )
        init_pred_scores = mo.to_dense(model.predicted_user_item_scores.copy())
        model.run(1)

        # assert new scores have changed
        trained_preds = mo.to_dense(model.predicted_user_item_scores.copy())
        with pytest.raises(AssertionError):
            test_helpers.assert_equal_arrays(init_pred_scores, trained_preds)

        # assert that recommendations are now "perfect"
        # every user should be recommended the item that
        # the person they are "following" interacted with
        model.num_items_per_iter = 1
        recommendations = model.recommend()
        correct_rec = np.array([[1], [2], [3], [4], [0]])
        test_helpers.assert_equal_arrays(recommendations, correct_rec)

        # ensure no errors when variuos arguments are sparse
        model = SocialFiltering(
            user_representation=social_network.copy(),
            num_items_per_iter=num_items,
        )
        model.run(1)

        model = SocialFiltering(
            actual_item_representation=items.copy(),
            actual_user_representation=users.copy(),
            num_items_per_iter=num_items,
        )
        model.run(1)

    def test_creator_items(self):
        users = np.random.randint(10, size=(100, 10))
        items = np.random.randint(2, size=(10, 100))
        creator_profiles = Creators(
            np.random.uniform(size=(50, 10)), creation_probability=1.0
        )  # 50 creator profiles
        sf = SocialFiltering(
            actual_user_representation=users,
            actual_item_representation=items,
            creators=creator_profiles,
        )
        sf.run(1, repeated_items=True)
        assert sf.items.num_items == 150  # 50 new items
        assert sf.items.num_attrs == 10  # 10 true items
        assert sf.items_hat.num_items == 150
        assert sf.items_hat.num_attrs == 100  # 100 users
        assert sf.users.actual_user_scores.num_users == 100
        assert sf.users.actual_user_scores.num_items == 150

    def test_new_users(self):
        # network initialized with 5 users
        # but in reality, there will be 10 total users
        num_users = 10
        num_items = 10
        users = np.eye(num_users)
        items = np.eye(num_items)
        social_network = np.roll(users, 1, axis=1)  # every user i is connected to (i+1) % 10

        first_users = users[:5, :].copy()
        first_network = social_network[:5, :5]
        model = SocialFiltering(
            user_representation=first_network,
            actual_item_representation=items,
            actual_user_representation=first_users,
            num_items_per_iter=10,
        )

        model.run(1, repeated_items=True)
        second_users = users[5:, :]
        model.add_users(second_users, social_graph=social_network)
        # should be a total of 10 users
        assert model.num_users == 10
        assert model.users.num_users == 10
        assert model.users_hat.num_users == 10
        assert model.users.actual_user_scores.num_users == 10
        # assert new users are represented as zeros
        test_helpers.assert_equal_arrays(social_network, model.users_hat.value)
        assert model.items_hat.value.sum() == 5.0
        model.run(1, repeated_items=True)
        # the first iteration should have yielded
        # 5 interactions, the second should yield another 10
        assert model.items_hat.value.sum() == 15.0
