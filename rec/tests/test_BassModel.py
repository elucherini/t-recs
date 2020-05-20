import test_utils
import numpy as np
from rec.models import BassModel
import pytest

class TestBassModel:
    def test_default(self):
        s = BassModel()
        test_utils.assert_correct_num_users(s.num_users, s, s.user_profiles.shape[0])
        test_utils.assert_correct_num_users(s.num_users, s, s.user_profiles.shape[1])
        test_utils.assert_correct_num_items(s.num_items, s, s.item_attributes.shape[1])
        test_utils.assert_not_none(s.predicted_scores)
        # did not set seed, show random behavior
        s1 = BassModel()

        with pytest.raises(AssertionError):
            test_utils.assert_equal_arrays(s.user_profiles, s1.user_profiles)

    def test_arguments(self, items=1, users=5):
        if items is None:
            items = 1
        if users is None:
            users = np.random.randint(1, 100)
        s = BassModel(num_users=users, num_items=items)
        test_utils.assert_correct_num_users(users, s, s.user_profiles.shape[0])
        test_utils.assert_correct_num_users(users, s, s.user_profiles.shape[1])
        test_utils.assert_correct_num_items(items, s, s.item_attributes.shape[1])
        test_utils.assert_not_none(s.predicted_scores)
        # did not set seed, show random behavior
        s1 = BassModel(num_users=users, num_items=items)

        with pytest.raises(AssertionError):
            test_utils.assert_equal_arrays(s.user_profiles, s1.user_profiles)

    def test_partial_arguments(self, items=1, users=5):
        if items is None:
            items = 1
        if users is None:
            users = np.random.randint(1, 100)
        # init with partially given arguments
        s = BassModel(num_users=users)
        test_utils.assert_correct_num_users(users, s, s.user_profiles.shape[0])
        test_utils.assert_correct_num_users(users, s, s.user_profiles.shape[1])
        test_utils.assert_correct_num_items(s.num_items, s, s.item_attributes.shape[1])
        test_utils.assert_not_none(s.predicted_scores)
        s = BassModel(num_items=items)
        test_utils.assert_correct_num_users(s.num_users, s, s.user_profiles.shape[0])
        test_utils.assert_correct_num_users(s.num_users, s, s.user_profiles.shape[1])
        test_utils.assert_correct_num_items(items, s, s.item_attributes.shape[1])
        test_utils.assert_not_none(s.predicted_scores)

        # did not set seed, show random behavior
        s1 = BassModel(num_users=users)

        with pytest.raises(AssertionError):
            test_utils.assert_equal_arrays(s.user_profiles, s1.user_profiles)
        s1 = BassModel(num_items=items)

        with pytest.raises(AssertionError):
            test_utils.assert_equal_arrays(s.user_profiles, s1.user_profiles)

    def test_representations(self, item_repr=None, user_repr=None):
        if item_repr is None:
            items = np.random.randint(1,1000)
            item_repr = np.random.random(size=(1,1))
        if user_repr is None or user_repr.shape[0] != user_repr.shape[1]:
            users = np.random.randint(1,100)
            user_repr = np.random.randint(2, size=(users, users))
        # test item representation
        s = BassModel(item_representation=item_repr)
        test_utils.assert_correct_num_users(s.num_users, s, s.user_profiles.shape[0])
        test_utils.assert_correct_num_users(s.num_users, s, s.user_profiles.shape[1])
        test_utils.assert_correct_num_items(item_repr.shape[1], s, s.item_attributes.shape[1])
        test_utils.assert_equal_arrays(item_repr, s.item_attributes)
        test_utils.assert_not_none(s.predicted_scores)

        # test user representation
        s = BassModel(user_representation=user_repr)
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
        s = BassModel(user_representation=user_repr, item_representation=item_repr)
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

        # did not set seed, show random behavior
        s1 = BassModel(item_representation=item_repr)

        with pytest.raises(AssertionError):
            test_utils.assert_equal_arrays(s.user_profiles, s1.user_profiles)

    def test_wrong_representations(self, bad_user_repr=None):
        if bad_user_repr is None or bad_user_repr.shape[0] == bad_user_repr.shape[1]:
            # bad_user_repr should not be a square matrix
            users = np.random.randint(1, 100)
            bad_user_repr = np.random.randint(2, size=(users + 2, users))
        with pytest.raises(ValueError):
            s = BassModel(user_representation=bad_user_repr)

    def test_additional_params(self, num_items_per_iter=None, num_new_items=None):
        # these are currently meaningless but at least it should not break
        if num_items_per_iter is None:
            # TODO vary parameter
            num_items_per_iter = 1#np.random.randint(5, 100)
        if num_new_items is None:
            num_new_items = np.random.randint(20, 400)
        s = BassModel(verbose=False, num_items_per_iter=num_items_per_iter,
                      num_new_items=num_new_items)
        assert(num_items_per_iter == s.num_items_per_iter)
        assert(num_new_items == s.num_new_items)
        # also check other params
        test_utils.assert_not_none(s.predicted_scores)
        test_utils.assert_correct_num_users(s.num_users, s, s.user_profiles.shape[0])
        test_utils.assert_correct_num_users(s.num_users, s, s.user_profiles.shape[1])
        test_utils.assert_correct_num_items(s.num_items, s, s.item_attributes.shape[1])

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
        while(user1 == user2):
            user1 = np.random.randint(s.num_users)
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

    def test_seeding(self, seed=None, items=None, users=None):
        if seed is None:
            seed = np.random.randint(100000)
        s1 = BassModel(seed=seed)
        s2 = BassModel(seed=seed)
        test_utils.assert_equal_arrays(s1.item_attributes, s2.item_attributes)
        test_utils.assert_equal_arrays(s1.user_profiles, s2.user_profiles)
        s1.run(timesteps=5)
        s2.run(timesteps=5)
        # check that measurements are the same
        meas1 = s1.get_measurements()
        meas2 = s2.get_measurements()
        systate1 = s1.get_system_state()
        systate2 = s2.get_system_state()
        test_utils.assert_equal_measurements(meas1, meas2)
        test_utils.assert_equal_system_state(systate1, systate2)

        if items is None:
            items = np.random.randint(1,1000)
        if users is None:
            users = np.random.randint(1,100)
        s1 = BassModel(seed=seed, num_users=users, num_items=items)
        s2 = BassModel(seed=seed, num_users=users, num_items=items)
        test_utils.assert_equal_arrays(s1.item_attributes, s2.item_attributes)
        test_utils.assert_equal_arrays(s1.user_profiles, s2.user_profiles)
        s1.run(timesteps=5)
        s2.run(timesteps=5)
        # check that measurements are the same
        meas1 = s1.get_measurements()
        meas2 = s2.get_measurements()
        test_utils.assert_equal_measurements(meas1, meas2)
        systate1 = s1.get_system_state()
        systate2 = s2.get_system_state()
        test_utils.assert_equal_system_state(systate1, systate2)

'''

'''
# --------------------------------------------------------- #
# --------------------------------------------------------- #
'''
if __name__ == '__main__':
    items = 3
    attr = 5
    users = 6
    logging.basicConfig(level=logging.INFO)

    # Define modules and how they map to test functions
    choices = ['content', 'user_scores', 'social', 'sir']
    choice_mapping = {'content': ContentFiltering_test,
                        'user_scores': ActualUserScores_test,
                        'social': SocialFiltering_test,
                        'sir': SIR_test}

    # Initialize parser and parse arguments
    parser = argparse.ArgumentParser(description='Test/debug recsys')
    parser.add_argument('--debug', '-d', choices=choices, required=True,
                        action='store', help='Decide on module to debug',
                        nargs='+')

    args = parser.parse_args()

    # For each module to test
    for module_name in args.debug:
        logger.info("# ------------------ #")
        logger.info('TESTING %s' % str(module_name.upper()))
        logger.info('# ------------------ #\n')
        try:
            # Use argument to find and run test function
            ret = choice_mapping[module_name](items, attr, users)
        except Exception as E:
            test_return(ret)
            raise E
        else:
            test_return(ret)
'''
