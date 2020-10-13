import test_helpers
import numpy as np
from trecs.models import SocialFiltering
import pytest


class TestSocialFiltering:
    def test_default(self):
        s = SocialFiltering()
        test_helpers.assert_correct_num_users(s.num_users, s, s.users_hat.shape[0])
        test_helpers.assert_correct_num_users(s.num_users, s, s.users_hat.shape[1])
        test_helpers.assert_correct_num_items(s.num_items, s, s.items_hat.shape[1])
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
        s = SocialFiltering(num_users=users, num_items=items)
        test_helpers.assert_correct_num_users(users, s, s.users_hat.shape[0])
        test_helpers.assert_correct_num_users(users, s, s.users_hat.shape[1])
        test_helpers.assert_correct_num_items(items, s, s.items_hat.shape[1])
        test_helpers.assert_not_none(s.predicted_scores)
        # did not set seed, show random behavior
        s1 = SocialFiltering(num_users=users, num_items=items)

        with pytest.raises(AssertionError):
            test_helpers.assert_equal_arrays(s.users_hat, s1.users_hat)

    def test_partial_arguments(self, items=10, users=5):
        if items is None:
            items = np.random.randint(10, 1000)
        if users is None:
            users = np.random.randint(10, 100)
        # init with partially given arguments
        s = SocialFiltering(num_users=users)
        test_helpers.assert_correct_num_users(users, s, s.users_hat.shape[0])
        test_helpers.assert_correct_num_users(users, s, s.users_hat.shape[1])
        test_helpers.assert_correct_num_items(s.num_items, s, s.items_hat.shape[1])
        test_helpers.assert_not_none(s.predicted_scores)

        # did not set seed, show random behavior
        s1 = SocialFiltering(num_users=users)

        with pytest.raises(AssertionError):
            test_helpers.assert_equal_arrays(s.users_hat, s1.users_hat)

        s = SocialFiltering(num_items=items)
        test_helpers.assert_correct_num_users(s.num_users, s, s.users_hat.shape[0])
        test_helpers.assert_correct_num_users(s.num_users, s, s.users_hat.shape[1])
        test_helpers.assert_correct_num_items(items, s, s.items_hat.shape[1])
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
        test_helpers.assert_correct_num_users(s.num_users, s, s.users_hat.shape[0])
        test_helpers.assert_correct_num_users(s.num_users, s, s.users_hat.shape[1])
        test_helpers.assert_correct_num_items(item_repr.shape[1], s, s.items_hat.shape[1])
        test_helpers.assert_equal_arrays(item_repr, s.items_hat)
        test_helpers.assert_not_none(s.predicted_scores)

        # did not set seed, show random behavior
        s1 = SocialFiltering(item_representation=item_repr)

        with pytest.raises(AssertionError):
            test_helpers.assert_equal_arrays(s.users_hat, s1.users_hat)

        # test user representation
        s = SocialFiltering(user_representation=user_repr)
        test_helpers.assert_correct_num_users(user_repr.shape[0], s, s.users_hat.shape[0])
        test_helpers.assert_correct_num_users(user_repr.shape[0], s, s.users_hat.shape[1])
        test_helpers.assert_correct_num_users(user_repr.shape[1], s, s.users_hat.shape[0])
        test_helpers.assert_correct_num_users(user_repr.shape[1], s, s.users_hat.shape[1])
        test_helpers.assert_correct_num_items(s.num_items, s, s.items_hat.shape[1])
        test_helpers.assert_equal_arrays(user_repr, s.users_hat)
        test_helpers.assert_not_none(s.predicted_scores)

        # test item and user representations
        s = SocialFiltering(user_representation=user_repr, item_representation=item_repr)
        test_helpers.assert_correct_num_users(user_repr.shape[0], s, s.users_hat.shape[0])
        test_helpers.assert_correct_num_users(user_repr.shape[0], s, s.users_hat.shape[1])
        test_helpers.assert_correct_num_users(user_repr.shape[1], s, s.users_hat.shape[0])
        test_helpers.assert_correct_num_users(user_repr.shape[1], s, s.users_hat.shape[1])
        test_helpers.assert_correct_num_items(item_repr.shape[1], s, s.items_hat.shape[1])
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
        test_helpers.assert_correct_num_users(s.num_users, s, s.users_hat.shape[0])
        test_helpers.assert_correct_num_users(s.num_users, s, s.users_hat.shape[1])
        test_helpers.assert_correct_num_items(s.num_items, s, s.items_hat.shape[1])
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
        test_helpers.assert_social_graph_not_following(s.users_hat, user1, user2)
        test_helpers.assert_social_graph_not_following(s.users_hat, user2, user1)
        # test follow
        s.follow(user1, user2)
        test_helpers.assert_social_graph_following(s.users_hat, user1, user2)
        test_helpers.assert_social_graph_not_following(s.users_hat, user2, user1)
        # test follow again -- nothing should change
        s.follow(user1, user2)
        test_helpers.assert_social_graph_following(s.users_hat, user1, user2)
        test_helpers.assert_social_graph_not_following(s.users_hat, user2, user1)
        # test unfollow
        s.unfollow(user1, user2)
        test_helpers.assert_social_graph_not_following(s.users_hat, user1, user2)
        test_helpers.assert_social_graph_not_following(s.users_hat, user2, user1)
        # test unfollow again -- nothing should change
        s.unfollow(user1, user2)
        test_helpers.assert_social_graph_not_following(s.users_hat, user1, user2)
        test_helpers.assert_social_graph_not_following(s.users_hat, user2, user1)

        # test friending
        s.add_friends(user1, user2)
        test_helpers.assert_social_graph_following(s.users_hat, user1, user2)
        test_helpers.assert_social_graph_following(s.users_hat, user2, user1)
        # test friending again -- nothing should change
        s.add_friends(user2, user1)
        test_helpers.assert_social_graph_following(s.users_hat, user1, user2)
        test_helpers.assert_social_graph_following(s.users_hat, user2, user1)
        # test unfriending
        s.remove_friends(user1, user2)
        test_helpers.assert_social_graph_not_following(s.users_hat, user1, user2)
        test_helpers.assert_social_graph_not_following(s.users_hat, user2, user1)
        # test unfriending again -- nothing should change
        s.remove_friends(user1, user2)
        test_helpers.assert_social_graph_not_following(s.users_hat, user1, user2)
        test_helpers.assert_social_graph_not_following(s.users_hat, user2, user1)

    def test_seeding(self, seed=None, items=None, users=None):
        if seed is None:
            seed = np.random.randint(100000)
        s1 = SocialFiltering(seed=seed)
        s2 = SocialFiltering(seed=seed)
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
            items = np.random.randint(1, 1000)
        if users is None:
            users = np.random.randint(1, 100)
        s1 = SocialFiltering(seed=seed, num_users=users, num_items=items)
        s2 = SocialFiltering(seed=seed, num_users=users, num_items=items)
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
