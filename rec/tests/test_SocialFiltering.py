import test_utils
import numpy as np
from rec import SocialFiltering
import pytest

class TestSocialFiltering:
    def test_default(self):
        s = SocialFiltering()
        test_utils.assert_correct_num_users(s.num_users, s, s.user_profiles.shape[0])
        test_utils.assert_correct_num_users(s.num_users, s, s.user_profiles.shape[1])
        test_utils.assert_correct_num_items(s.num_items, s, s.item_attributes.shape[1])
        test_utils.assert_not_none(s.predicted_scores)

    def test_arguments(self, items=10, users=5):
        if items is None:
            items = np.random.randint(1000)
        if users is None:
            users = np.random.randint(100)
        s = SocialFiltering(num_users=users, num_items=items)
        test_utils.assert_correct_num_users(users, s, s.user_profiles.shape[0])
        test_utils.assert_correct_num_users(users, s, s.user_profiles.shape[1])
        test_utils.assert_correct_num_items(items, s, s.item_attributes.shape[1])
        test_utils.assert_not_none(s.predicted_scores)

    def test_partial_arguments(self, items=10, users=5):
        if items is None:
            items = np.random.randint(1000)
        if users is None:
            users = np.random.randint(100)
        # init with partially given arguments
        s = SocialFiltering(num_users=users)
        test_utils.assert_correct_num_users(users, s, s.user_profiles.shape[0])
        test_utils.assert_correct_num_users(users, s, s.user_profiles.shape[1])
        test_utils.assert_correct_num_items(s.num_items, s, s.item_attributes.shape[1])
        test_utils.assert_not_none(s.predicted_scores)
        s = SocialFiltering(num_items=items)
        test_utils.assert_correct_num_users(s.num_users, s, s.user_profiles.shape[0])
        test_utils.assert_correct_num_users(s.num_users, s, s.user_profiles.shape[1])
        test_utils.assert_correct_num_items(items, s, s.item_attributes.shape[1])
        test_utils.assert_not_none(s.predicted_scores)

    def test_representations(self, item_repr=None,
                             user_repr=None):
        if item_repr is None:
            items = np.random.randint(1000)
            users = (user_repr.shape[0] if user_repr is not None
                     else np.random.randint(100))
            item_repr = np.random.random(size=(users,items))
        if user_repr is None or user_repr.shape[0] != user_repr.shape[1]:
            users = item_repr.shape[0]
            user_repr = np.random.randint(2, size=(users, users))
        # test item representation
        s = SocialFiltering(item_representation=item_repr)
        test_utils.assert_correct_num_users(s.num_users, s, s.user_profiles.shape[0])
        test_utils.assert_correct_num_users(s.num_users, s, s.user_profiles.shape[1])
        test_utils.assert_correct_num_items(item_repr.shape[1], s, s.item_attributes.shape[1])
        test_utils.assert_equal_arrays(item_repr, s.item_attributes)
        test_utils.assert_not_none(s.predicted_scores)

        # test user representation
        s = SocialFiltering(user_representation=user_repr)
        test_utils.assert_correct_num_users(user_repr.shape[0], s,
                                            s.user_profiles.shape[0])
        test_utils.assert_correct_num_users(user_repr.shape[0], s,
                                            s.user_profiles.shape[1])
        test_utils.assert_correct_num_users(user_repr.shape[1], s,
                                            s.user_profiles.shape[0])
        test_utils.assert_correct_num_users(user_repr.shape[1], s,
                                            s.user_profiles.shape[1])
        test_utils.assert_correct_num_items(s.num_items, s, s.item_attributes.shape[1])
        test_utils.assert_equal_arrays(user_repr, s.user_profiles)
        test_utils.assert_not_none(s.predicted_scores)

        # test item and user representations
        s = SocialFiltering(user_representation=user_repr, item_representation=item_repr)
        test_utils.assert_correct_num_users(user_repr.shape[0], s,
                                 s.user_profiles.shape[0])
        test_utils.assert_correct_num_users(user_repr.shape[0], s,
                                 s.user_profiles.shape[1])
        test_utils.assert_correct_num_users(user_repr.shape[1], s,
                                 s.user_profiles.shape[0])
        test_utils.assert_correct_num_users(user_repr.shape[1], s,
                                 s.user_profiles.shape[1])
        test_utils.assert_correct_num_items(item_repr.shape[1], s, s.item_attributes.shape[1])
        test_utils.assert_equal_arrays(user_repr, s.user_profiles)
        test_utils.assert_equal_arrays(item_repr, s.item_attributes)
        test_utils.assert_not_none(s.predicted_scores)

    def test_wrong_representations(self, bad_user_repr=None):
        if bad_user_repr is None or bad_user_repr.shape[0] == bad_user_repr.shape[1]:
            # bad_user_repr should not be a square matrix
            users = np.random.randint(100)
            bad_user_repr = np.random.randint(2, size=(users + 2, users))
        # this should throw an exception
        with pytest.raises(ValueError):
            s = SocialFiltering(user_representation=bad_user_repr)

    def test_additional_params(self, num_items_per_iter=None, num_new_items=None):
        if num_items_per_iter is None:
            num_items_per_iter = np.random.randint(5, 100)
        if num_new_items is None:
            num_new_items = np.random.randint(20, 400)
        s = SocialFiltering(verbose=False, num_items_per_iter=num_items_per_iter,
                      num_new_items=num_new_items)
        assert(num_items_per_iter == s.num_items_per_iter)
        assert(num_new_items == s.num_new_items)
        # also check other params
        test_utils.assert_correct_num_users(s.num_users, s, s.user_profiles.shape[0])
        test_utils.assert_correct_num_users(s.num_users, s, s.user_profiles.shape[1])
        test_utils.assert_correct_num_items(s.num_items, s, s.item_attributes.shape[1])
        test_utils.assert_not_none(s.predicted_scores)

    def test_social_graph(self, user_repr=None, user1=None, user2=None):
        if user_repr is None or user_repr.shape[0] != user_repr.shape[1]:
            users = np.random.randint(100)
            user_repr = np.zeros((users, users))
        s = SocialFiltering(user_representation=user_repr)
        if user1 is None:
            user1 = np.random.randint(s.num_users)
        if user2 is None:
            user2 = np.random.randint(s.num_users)
        # users must be different
        while(user1 == user2):
            user1 = np.random.randint(s.num_users)
            user2 = np.random.randint(s.num_users)
        # test current graph
        test_utils.assert_social_graph_not_following(s.user_profiles, user1, user2)
        test_utils.assert_social_graph_not_following(s.user_profiles, user2, user1)
        # test follow
        s.follow(user1, user2)
        test_utils.assert_social_graph_following(s.user_profiles, user1, user2)
        test_utils.assert_social_graph_not_following(s.user_profiles, user2, user1)
        # test follow again -- nothing should change
        s.follow(user1, user2)
        test_utils.assert_social_graph_following(s.user_profiles, user1, user2)
        test_utils.assert_social_graph_not_following(s.user_profiles, user2, user1)
        # test unfollow
        s.unfollow(user1, user2)
        test_utils.assert_social_graph_not_following(s.user_profiles, user1, user2)
        test_utils.assert_social_graph_not_following(s.user_profiles, user2, user1)
        # test unfollow again -- nothing should change
        s.unfollow(user1, user2)
        test_utils.assert_social_graph_not_following(s.user_profiles, user1, user2)
        test_utils.assert_social_graph_not_following(s.user_profiles, user2, user1)

        # test friending
        s.add_friends(user1, user2)
        test_utils.assert_social_graph_following(s.user_profiles, user1, user2)
        test_utils.assert_social_graph_following(s.user_profiles, user2, user1)
        # test friending again -- nothing should change
        s.add_friends(user2, user1)
        test_utils.assert_social_graph_following(s.user_profiles, user1, user2)
        test_utils.assert_social_graph_following(s.user_profiles, user2, user1)
        # test unfriending
        s.remove_friends(user1, user2)
        test_utils.assert_social_graph_not_following(s.user_profiles, user1, user2)
        test_utils.assert_social_graph_not_following(s.user_profiles, user2, user1)
        # test unfriending again -- nothing should change
        s.remove_friends(user1, user2)
        test_utils.assert_social_graph_not_following(s.user_profiles, user1, user2)
        test_utils.assert_social_graph_not_following(s.user_profiles, user2, user1)
