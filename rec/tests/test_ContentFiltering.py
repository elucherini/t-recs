from rec.models import ContentFiltering
import numpy as np
import pytest
import test_utils

class TestContentFiltering:
    def test_default(self):
        c = ContentFiltering()
        test_utils.assert_correct_num_users(c.num_users, c,
                                            c.user_profiles.shape[0])
        test_utils.assert_correct_num_items(c.num_items, c,
                                            c.item_attributes.shape[1])
        test_utils.assert_correct_size_generic(c.num_attributes, c.num_attributes,
                                           c.user_profiles.shape[1])
        test_utils.assert_correct_size_generic(c.num_attributes, c.num_attributes,
                                           c.item_attributes.shape[0])
        test_utils.assert_not_none(c.predicted_scores)

        # did not set seed, show random behavior
        c1 = ContentFiltering()

        with pytest.raises(AssertionError):
            test_utils.assert_equal_arrays(c.item_attributes, c1.item_attributes)

    def test_arguments(self, items=None, attr=None, users=None):
        if items is None:
            items = np.random.randint(1,1000)
        if users is None:
            users = np.random.randint(1,100)
        if attr is None:
            attr = np.random.randint(1,100)
        # init with given arguments
        c = ContentFiltering(num_users=users, num_items=items, num_attributes=attr)
        test_utils.assert_correct_num_users(users, c, c.user_profiles.shape[0])
        test_utils.assert_correct_num_items(items, c, c.item_attributes.shape[1])
        test_utils.assert_correct_size_generic(attr, c.num_attributes,
                                               c.item_attributes.shape[0])
        test_utils.assert_correct_size_generic(attr, c.num_attributes,
                                               c.user_profiles.shape[1])
        test_utils.assert_not_none(c.predicted_scores)

        # did not set seed, show random behavior
        c1 = ContentFiltering(num_users=users, num_items=items, num_attributes=attr)

        with pytest.raises(AssertionError):
            test_utils.assert_equal_arrays(c.item_attributes, c1.item_attributes)

    def test_partial_arguments(self, items=None, users=None, attr=None):
        # init with partially given arguments
        if items is None:
            items = np.random.randint(1,1000)
        if users is None:
            users = np.random.randint(1,100)
        if attr is None:
            attr = np.random.randint(1,100)

        c = ContentFiltering(num_users=users)
        test_utils.assert_correct_num_users(users, c, c.user_profiles.shape[0])
        test_utils.assert_correct_num_items(c.num_items, c, c.item_attributes.shape[1])
        test_utils.assert_correct_size_generic(c.num_attributes, c.num_attributes,
                                               c.item_attributes.shape[0])
        test_utils.assert_correct_size_generic(c.num_attributes, c.num_attributes,
                                               c.user_profiles.shape[1])
        test_utils.assert_not_none(c.predicted_scores)

        # did not set seed, show random behavior
        c1 = ContentFiltering(num_users=users)

        with pytest.raises(AssertionError):
            test_utils.assert_equal_arrays(c.item_attributes, c1.item_attributes)

        c = ContentFiltering(num_items=items)
        test_utils.assert_correct_num_users(c.num_users, c, c.user_profiles.shape[0])
        test_utils.assert_correct_num_items(items, c, c.item_attributes.shape[1])
        test_utils.assert_correct_size_generic(c.num_attributes, c.num_attributes,
                                               c.item_attributes.shape[0])
        test_utils.assert_correct_size_generic(c.num_attributes, c.num_attributes,
                                               c.user_profiles.shape[1])
        test_utils.assert_not_none(c.predicted_scores)

        # did not set seed, show random behavior
        c1 = ContentFiltering(num_items=items)

        with pytest.raises(AssertionError):
            test_utils.assert_equal_arrays(c.item_attributes, c1.item_attributes)

        c = ContentFiltering(num_attributes=attr)
        test_utils.assert_correct_num_users(c.num_users, c, c.user_profiles.shape[0])
        test_utils.assert_correct_num_items(c.num_items, c, c.item_attributes.shape[1])
        test_utils.assert_correct_size_generic(attr, c.num_attributes,
                                               c.item_attributes.shape[0])
        test_utils.assert_correct_size_generic(attr, c.num_attributes,
                                               c.user_profiles.shape[1])
        test_utils.assert_not_none(c.predicted_scores)

        # did not set seed, show random behavior
        c1 = ContentFiltering(num_attributes=attr)

        with pytest.raises(AssertionError):
            test_utils.assert_equal_arrays(c.item_attributes, c1.item_attributes)

        c = ContentFiltering(num_users=users, num_items=items)
        test_utils.assert_correct_num_users(users, c, c.user_profiles.shape[0])
        test_utils.assert_correct_num_items(items, c, c.item_attributes.shape[1])
        test_utils.assert_correct_size_generic(c.num_attributes, c.num_attributes,
                                               c.item_attributes.shape[0])
        test_utils.assert_correct_size_generic(c.num_attributes, c.num_attributes,
                                               c.user_profiles.shape[1])
        test_utils.assert_not_none(c.predicted_scores)

        # did not set seed, show random behavior
        c1 = ContentFiltering(num_users=users, num_items=items)

        with pytest.raises(AssertionError):
            test_utils.assert_equal_arrays(c.item_attributes, c1.item_attributes)

        c = ContentFiltering(num_users=users, num_attributes=attr)
        test_utils.assert_correct_num_users(users, c, c.user_profiles.shape[0])
        test_utils.assert_correct_num_items(c.num_items, c, c.item_attributes.shape[1])
        test_utils.assert_correct_size_generic(attr, c.num_attributes,
                                               c.item_attributes.shape[0])
        test_utils.assert_correct_size_generic(attr, c.num_attributes,
                                               c.user_profiles.shape[1])
        test_utils.assert_not_none(c.predicted_scores)

        # did not set seed, show random behavior
        c1 = ContentFiltering(num_users=users, num_attributes=attr)

        with pytest.raises(AssertionError):
            test_utils.assert_equal_arrays(c.item_attributes, c1.item_attributes)

        c = ContentFiltering(num_attributes=attr, num_items=items)
        test_utils.assert_correct_num_users(c.num_users, c, c.user_profiles.shape[0])
        test_utils.assert_correct_num_items(items, c, c.item_attributes.shape[1])
        test_utils.assert_correct_size_generic(attr, c.num_attributes,
                                               c.item_attributes.shape[0])
        test_utils.assert_correct_size_generic(attr, c.num_attributes,
                                               c.user_profiles.shape[1])
        test_utils.assert_not_none(c.predicted_scores)

        # did not set seed, show random behavior
        c1 = ContentFiltering(num_attributes=attr, num_items=items)

        with pytest.raises(AssertionError):
            test_utils.assert_equal_arrays(c.item_attributes, c1.item_attributes)


    def test_representations(self, item_repr=None, user_repr=None, bad_user_repr=None):
        if item_repr is None:
            items = np.random.randint(1000)
            attr = np.random.randint(10)
            item_repr = np.random.random(size=(attr,items))
        if user_repr is None or user_repr.shape[1] != item_repr.shape[0]:
            users = np.random.randint(100)
            user_repr = np.random.randint(10, size=(users, item_repr.shape[0]))

        c = ContentFiltering(item_representation=item_repr)
        test_utils.assert_correct_num_users(c.num_users, c, c.user_profiles.shape[0])
        test_utils.assert_correct_num_items(item_repr.shape[1], c,
                                            c.item_attributes.shape[1])
        test_utils.assert_correct_size_generic(item_repr.shape[0], c.num_attributes,
                                               c.item_attributes.shape[0])
        test_utils.assert_correct_size_generic(item_repr.shape[0], c.num_attributes,
                                               c.user_profiles.shape[1])
        test_utils.assert_equal_arrays(item_repr, c.item_attributes)
        test_utils.assert_not_none(c.predicted_scores)


        c = ContentFiltering(user_representation=user_repr)
        test_utils.assert_correct_num_users(user_repr.shape[0], c,
                                            c.user_profiles.shape[0])
        test_utils.assert_correct_num_items(c.num_items, c,
                                            c.item_attributes.shape[1])
        test_utils.assert_correct_size_generic(user_repr.shape[1], c.num_attributes,
                                               c.item_attributes.shape[0])
        test_utils.assert_correct_size_generic(user_repr.shape[1], c.num_attributes,
                                               c.user_profiles.shape[1])
        test_utils.assert_equal_arrays(user_repr, c.user_profiles)
        test_utils.assert_not_none(c.predicted_scores)

        # did not set seed, show random behavior
        c1 = ContentFiltering(user_representation=user_repr)

        #with pytest.raises(AssertionError):
        #    test_utils.assert_equal_arrays(c.user_profiles, c1.user_profiles)
        with pytest.raises(AssertionError):
            test_utils.assert_equal_arrays(c.item_attributes, c1.item_attributes)
        #with pytest.raises(AssertionError):
        #    test_utils.assert_equal_arrays(c.num_attributes, c1.num_attributes)


        c = ContentFiltering(user_representation=user_repr,
                             item_representation=item_repr)
        test_utils.assert_correct_num_users(user_repr.shape[0], c,
                                            c.user_profiles.shape[0])
        test_utils.assert_correct_num_items(item_repr.shape[1], c,
                                            c.item_attributes.shape[1])
        test_utils.assert_correct_size_generic(user_repr.shape[1], c.num_attributes,
                                               c.item_attributes.shape[0])
        test_utils.assert_correct_size_generic(user_repr.shape[1], c.num_attributes,
                                               c.user_profiles.shape[1])
        test_utils.assert_correct_size_generic(item_repr.shape[0], c.num_attributes,
                                               c.item_attributes.shape[0])
        test_utils.assert_correct_size_generic(item_repr.shape[0], c.num_attributes,
                                               c.user_profiles.shape[1])
        test_utils.assert_equal_arrays(user_repr, c.user_profiles)
        test_utils.assert_equal_arrays(item_repr, c.item_attributes)
        test_utils.assert_not_none(c.predicted_scores)

    def test_wrong_representation(self, user_repr=None, item_repr=None,
                                  bad_user_repr=None, bad_item_repr=None):
        if item_repr is None:
            items = np.random.randint(1000)
            attr = np.random.randint(10)
            item_repr = np.random.random(size=(attr,items))
        if user_repr is None or user_repr.shape[1] != item_repr.shape[0]:
            users = np.random.randint(100)
            user_repr = np.random.randint(10, size=(users, item_repr.shape[0]))

        if bad_user_repr is None or bad_user_repr.shape[1] == item_repr.shape[0]:
            # |A| shouldn't match item_repr.shape[0]
            bad_user_repr = np.random.randint(10, size=(user_repr.shape[0], user_repr.shape[1] + 2))
        if bad_item_repr is None or bad_item_repr.shape[0] == user_repr.shape[1]:
            # |A| shouldn't match user_repr.shape[1]
            bad_item_repr = np.random.random(size=(item_repr.shape[0] + 1, item_repr.shape[1]))

        with pytest.raises(ValueError):
            c = ContentFiltering(user_representation=bad_user_repr,
                                 item_representation=item_repr)
        with pytest.raises(ValueError):
            c = ContentFiltering(user_representation=user_repr,
                                 item_representation=bad_item_repr)

    def test_additional_params(self, num_items_per_iter=None, num_new_items=None):
        if num_items_per_iter is None:
            num_items_per_iter = np.random.randint(5, 100)
        if num_new_items is None:
            num_new_items = np.random.randint(20, 400)

        c = ContentFiltering(verbose=False, num_items_per_iter=num_items_per_iter,
                             num_new_items=num_new_items)
        assert(num_items_per_iter == c.num_items_per_iter)
        assert(num_new_items == c.num_new_items)
        # also check other params
        test_utils.assert_correct_num_users(c.num_users, c,
                                            c.user_profiles.shape[0])
        test_utils.assert_correct_num_items(c.num_items, c,
                                            c.item_attributes.shape[1])
        test_utils.assert_correct_size_generic(c.num_attributes, c.num_attributes,
                                           c.user_profiles.shape[1])
        test_utils.assert_correct_size_generic(c.num_attributes, c.num_attributes,
                                           c.item_attributes.shape[0])
        test_utils.assert_not_none(c.predicted_scores)


    def test_seeding(self, seed=None, items=None, users=None):
        if seed is None:
            seed = np.random.randint(100000)
        s1 = ContentFiltering(seed=seed)
        s2 = ContentFiltering(seed=seed)
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

        if items is None:
            items = np.random.randint(1,1000)
        if users is None:
            users = np.random.randint(1,100)
        s1 = ContentFiltering(seed=seed, num_users=users, num_items=items)
        s2 = ContentFiltering(seed=seed, num_users=users, num_items=items)
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

