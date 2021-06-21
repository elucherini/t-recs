from trecs.metrics.measurement import MSEMeasurement
import test_helpers
import numpy as np
from scipy.sparse import csr_matrix
from trecs.models import BassModel
import pytest


class TestBassModel:
    def test_default(self):
        s = BassModel()
        test_helpers.assert_correct_num_users(s.num_users, s, s.users_hat.num_users)
        test_helpers.assert_correct_num_users(s.num_users, s, s.users_hat.num_attrs)
        test_helpers.assert_correct_num_items(s.num_items, s, s.items_hat.num_items)
        test_helpers.assert_not_none(s.predicted_scores)
        # did not set seed, show random behavior
        s1 = BassModel()

        with pytest.raises(AssertionError):
            test_helpers.assert_equal_arrays(s.users_hat.value, s1.users_hat)

    def test_arguments(self, items=1, users=5):
        if items is None:
            items = 1
        if users is None:
            users = np.random.randint(30, 100)
        s = BassModel(num_users=users, num_items=items, seed=100)
        test_helpers.assert_correct_num_users(users, s, s.users_hat.num_users)
        test_helpers.assert_correct_num_users(users, s, s.users_hat.num_attrs)
        test_helpers.assert_correct_num_items(items, s, s.items_hat.num_items)
        test_helpers.assert_not_none(s.predicted_scores)
        # show random behavior with different seeds
        s1 = BassModel(num_users=users, num_items=items, seed=101)

        with pytest.raises(AssertionError):
            test_helpers.assert_equal_arrays(s.users_hat.value, s1.users_hat)

    def test_partial_arguments(self, items=1, users=5):
        if items is None:
            items = 1
        if users is None:
            users = np.random.randint(1, 100)
        # init with partially given arguments
        s = BassModel(num_users=users)
        test_helpers.assert_correct_num_users(users, s, s.users_hat.num_users)
        test_helpers.assert_correct_num_users(users, s, s.users_hat.num_attrs)
        test_helpers.assert_correct_num_items(s.num_items, s, s.items_hat.num_items)
        test_helpers.assert_not_none(s.predicted_scores)
        s = BassModel(num_items=items)
        test_helpers.assert_correct_num_users(s.num_users, s, s.users_hat.num_users)
        test_helpers.assert_correct_num_users(s.num_users, s, s.users_hat.num_attrs)
        test_helpers.assert_correct_num_items(items, s, s.items_hat.num_items)
        test_helpers.assert_not_none(s.predicted_scores)

        # did not set seed, show random behavior
        s1 = BassModel(num_users=users)

        with pytest.raises(AssertionError):
            test_helpers.assert_equal_arrays(s.users_hat.value, s1.users_hat)
        s1 = BassModel(num_items=items)

        with pytest.raises(AssertionError):
            test_helpers.assert_equal_arrays(s.users_hat.value, s1.users_hat)

    def test_representations(self, item_repr=None, user_repr=None):
        if item_repr is None:
            items = np.random.randint(1, 1000)
            item_repr = np.random.random(size=(1, 1))
        if user_repr is None or user_repr.shape[0] != user_repr.shape[1]:
            users = np.random.randint(1, 100)
            user_repr = np.random.randint(2, size=(users, users))
        # test item representation
        s = BassModel(item_representation=item_repr)
        test_helpers.assert_correct_num_users(s.num_users, s, s.users_hat.num_users)
        test_helpers.assert_correct_num_users(s.num_users, s, s.users_hat.num_attrs)
        test_helpers.assert_correct_num_items(item_repr.shape[1], s, s.items_hat.num_items)
        test_helpers.assert_equal_arrays(item_repr, s.items_hat)
        test_helpers.assert_not_none(s.predicted_scores)

        # test user representation
        s = BassModel(user_representation=user_repr)
        test_helpers.assert_correct_num_users(user_repr.shape[0], s, s.users_hat.num_users)
        test_helpers.assert_correct_num_users(user_repr.shape[0], s, s.users_hat.num_attrs)
        test_helpers.assert_correct_num_users(user_repr.shape[1], s, s.users_hat.num_users)
        test_helpers.assert_correct_num_users(user_repr.shape[1], s, s.users_hat.num_attrs)
        test_helpers.assert_correct_num_items(s.num_items, s, s.items_hat.num_items)
        test_helpers.assert_equal_arrays(user_repr, s.users_hat)
        test_helpers.assert_not_none(s.predicted_scores)

        # test item and user representations
        s = BassModel(user_representation=user_repr, item_representation=item_repr)
        test_helpers.assert_correct_num_users(user_repr.shape[0], s, s.users_hat.num_users)
        test_helpers.assert_correct_num_users(user_repr.shape[0], s, s.users_hat.num_attrs)
        test_helpers.assert_correct_num_users(user_repr.shape[1], s, s.users_hat.num_users)
        test_helpers.assert_correct_num_users(user_repr.shape[1], s, s.users_hat.num_attrs)
        test_helpers.assert_correct_num_items(item_repr.shape[1], s, s.items_hat.num_items)
        test_helpers.assert_equal_arrays(user_repr, s.users_hat)
        test_helpers.assert_equal_arrays(item_repr, s.items_hat)
        test_helpers.assert_not_none(s.predicted_scores)

        # did not set seed, show random behavior
        s1 = BassModel(item_representation=item_repr)

        with pytest.raises(AssertionError):
            test_helpers.assert_equal_arrays(s.users_hat.value, s1.users_hat)

    def test_wrong_representations(self, bad_user_repr=None):
        if bad_user_repr is None or bad_user_repr.shape[0] == bad_user_repr.shape[1]:
            # bad_user_repr should not be a square matrix
            users = np.random.randint(1, 100)
            bad_user_repr = np.random.randint(2, size=(users + 2, users))
        with pytest.raises(ValueError):
            s = BassModel(user_representation=bad_user_repr)

    def test_additional_params(self, num_items_per_iter=None):
        # these are currently meaningless but at least it should not break
        if num_items_per_iter is None:
            num_items_per_iter = 1
        s = BassModel(verbose=False, num_items_per_iter=num_items_per_iter)
        assert num_items_per_iter == s.num_items_per_iter
        # also check other params
        test_helpers.assert_not_none(s.predicted_scores)
        test_helpers.assert_correct_num_users(s.num_users, s, s.users_hat.num_users)
        test_helpers.assert_correct_num_users(s.num_users, s, s.users_hat.num_attrs)
        test_helpers.assert_correct_num_items(s.num_items, s, s.items_hat.num_items)

    def test_social_graph(self, user_repr=None, user1=None, user2=None):
        if user_repr is None or user_repr.shape[0] != user_repr.shape[1]:
            users = np.random.randint(1, 100)
            user_repr = np.zeros((users, users))
        s = BassModel(user_representation=user_repr)
        if user1 is None:
            user1 = np.random.randint(s.num_users)
        if user2 is None:
            user2 = np.random.randint(s.num_users)
        # users must be different
        while user1 == user2:
            user1 = np.random.randint(s.num_users)
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
        s1 = BassModel(seed=seed, record_base_state=True)
        s1.add_metrics(MSEMeasurement())
        s2 = BassModel(seed=seed, record_base_state=True)
        s2.add_metrics(MSEMeasurement())
        test_helpers.assert_equal_arrays(s1.items_hat, s2.items_hat)
        test_helpers.assert_equal_arrays(s1.users_hat, s2.users_hat)
        s1.run(timesteps=5)
        s2.run(timesteps=5)
        # check that measurements are the same
        meas1 = s1.get_measurements()
        meas2 = s2.get_measurements()
        systate1 = s1.get_system_state()
        systate2 = s2.get_system_state()
        test_helpers.assert_equal_measurements(meas1, meas2)
        test_helpers.assert_equal_system_state(systate1, systate2)

        if items is None:
            items = np.random.randint(20, 1000)
        if users is None:
            users = np.random.randint(1, 100)
        s1 = BassModel(seed=seed, num_users=users, num_items=items, record_base_state=True)
        s1.add_metrics(MSEMeasurement())
        s2 = BassModel(seed=seed, num_users=users, num_items=items, record_base_state=True)
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

    def test_infections(self):
        num_users = 5
        user_rep = np.zeros((num_users, num_users))
        user_rep[1, 0] = 1  # user 1 is connected to user 0
        user_rep[0, 1] = 1  # user 1 is connected to user 0
        user_rep[2, 1] = 1  # user 2 is connected to user 1
        infection_state = np.zeros((num_users, 1))
        infection_state[0] = 1  # user 0 is infected at the outset
        item_rep = np.array(
            [[0.9999]]
        )  # combined with random seed, this should guarantee infection
        bass = BassModel(
            user_representation=user_rep,
            item_representation=item_rep,
            infection_state=infection_state,
            seed=1234,
        )
        bass.run(1)  # after 1st step, user 1 should be infected, and user 0 should be recovered
        correct_infections = np.array([-1, 1, 0, 0, 0]).reshape(-1, 1)
        test_helpers.assert_equal_arrays(infection_state, correct_infections)
        bass.run(
            1
        )  # after 2nd step, users 0 and 1 should be recovered, and user 2 should be infected
        correct_infections = np.array([-1, -1, 1, 0, 0]).reshape(-1, 1)
        test_helpers.assert_equal_arrays(infection_state, correct_infections)

        # test running to completion
        bass.run()
        correct_infections = np.array([-1, -1, -1, 0, 0]).reshape(-1, 1)
        test_helpers.assert_equal_arrays(infection_state, correct_infections)

    def test_sparse_matrix(self):
        num_users = 5
        user_rep = csr_matrix(np.zeros((num_users, num_users)))
        user_rep[1, 0] = 1  # user 1 is connected to user 0
        user_rep[0, 1] = 1  # user 1 is connected to user 0
        user_rep[2, 1] = 1  # user 2 is connected to user 1
        infection_state = np.zeros((num_users, 1))
        infection_state[0] = 1  # user 0 is infected at the outset
        item_rep = np.array(
            [[0.9999]]
        )  # combined with random seed, this should guarantee infection
        bass = BassModel(
            user_representation=user_rep,
            item_representation=item_rep,
            infection_state=infection_state,
            seed=132,
        )

        bass.run(1)  # after 1st step, user 1 should be infected, and user 0 should be recovered
        correct_infections = np.array([-1, 1, 0, 0, 0]).reshape(-1, 1)
        test_helpers.assert_equal_arrays(infection_state, correct_infections)
        bass.run(
            1
        )  # after 2nd step, users 0 and 1 should be recovered, and user 2 should be infected
        correct_infections = np.array([-1, -1, 1, 0, 0]).reshape(-1, 1)
        test_helpers.assert_equal_arrays(infection_state, correct_infections)

        # assert that the user representation is still sparse
        assert isinstance(bass.users_hat.value, csr_matrix)
