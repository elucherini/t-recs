from rec.models import ContentFiltering
from rec.components import Users
import numpy as np
import pytest
import test_helpers
from rec.utils import normalize_matrix


class TestContentFiltering:
    def test_default(self):
        c = ContentFiltering()
        test_helpers.assert_correct_num_users(c.num_users, c, c.users_hat.shape[0])
        test_helpers.assert_correct_num_items(c.num_items, c, c.items_hat.shape[1])
        test_helpers.assert_correct_size_generic(
            c.num_attributes, c.num_attributes, c.users_hat.shape[1]
        )
        test_helpers.assert_correct_size_generic(
            c.num_attributes, c.num_attributes, c.items_hat.shape[0]
        )
        test_helpers.assert_not_none(c.predicted_scores)

        # did not set seed, show random behavior
        c1 = ContentFiltering()

        with pytest.raises(AssertionError):
            test_helpers.assert_equal_arrays(c.items_hat, c1.items_hat)

    def test_arguments(self, items=None, attr=None, users=None):
        if items is None:
            items = np.random.randint(1, 1000)
        if users is None:
            users = np.random.randint(1, 100)
        if attr is None:
            attr = np.random.randint(1, 100)
        # init with given arguments
        c = ContentFiltering(num_users=users, num_items=items, num_attributes=attr)
        test_helpers.assert_correct_num_users(users, c, c.users_hat.shape[0])
        test_helpers.assert_correct_num_items(items, c, c.items_hat.shape[1])
        test_helpers.assert_correct_size_generic(
            attr, c.num_attributes, c.items_hat.shape[0]
        )
        test_helpers.assert_correct_size_generic(
            attr, c.num_attributes, c.users_hat.shape[1]
        )
        test_helpers.assert_not_none(c.predicted_scores)

        # did not set seed, show random behavior
        c1 = ContentFiltering(num_users=users, num_items=items, num_attributes=attr)

        with pytest.raises(AssertionError):
            test_helpers.assert_equal_arrays(c.items_hat, c1.items_hat)

    def test_partial_arguments(self, items=None, users=None, attr=None):
        # init with partially given arguments
        if items is None:
            items = np.random.randint(1, 1000)
        if users is None:
            users = np.random.randint(1, 100)
        if attr is None:
            attr = np.random.randint(1, 100)

        c = ContentFiltering(num_users=users)
        test_helpers.assert_correct_num_users(users, c, c.users_hat.shape[0])
        test_helpers.assert_correct_num_items(c.num_items, c, c.items_hat.shape[1])
        test_helpers.assert_correct_size_generic(
            c.num_attributes, c.num_attributes, c.items_hat.shape[0]
        )
        test_helpers.assert_correct_size_generic(
            c.num_attributes, c.num_attributes, c.users_hat.shape[1]
        )
        test_helpers.assert_not_none(c.predicted_scores)

        # did not set seed, show random behavior
        c1 = ContentFiltering(num_users=users)

        with pytest.raises(AssertionError):
            test_helpers.assert_equal_arrays(c.items_hat, c1.items_hat)

        c = ContentFiltering(num_items=items)
        test_helpers.assert_correct_num_users(c.num_users, c, c.users_hat.shape[0])
        test_helpers.assert_correct_num_items(items, c, c.items_hat.shape[1])
        test_helpers.assert_correct_size_generic(
            c.num_attributes, c.num_attributes, c.items_hat.shape[0]
        )
        test_helpers.assert_correct_size_generic(
            c.num_attributes, c.num_attributes, c.users_hat.shape[1]
        )
        test_helpers.assert_not_none(c.predicted_scores)

        # did not set seed, show random behavior
        c1 = ContentFiltering(num_items=items)

        with pytest.raises(AssertionError):
            test_helpers.assert_equal_arrays(c.items_hat, c1.items_hat)

        c = ContentFiltering(num_attributes=attr)
        test_helpers.assert_correct_num_users(c.num_users, c, c.users_hat.shape[0])
        test_helpers.assert_correct_num_items(c.num_items, c, c.items_hat.shape[1])
        test_helpers.assert_correct_size_generic(
            attr, c.num_attributes, c.items_hat.shape[0]
        )
        test_helpers.assert_correct_size_generic(
            attr, c.num_attributes, c.users_hat.shape[1]
        )
        test_helpers.assert_not_none(c.predicted_scores)

        # did not set seed, show random behavior
        c1 = ContentFiltering(num_attributes=attr)

        with pytest.raises(AssertionError):
            test_helpers.assert_equal_arrays(c.items_hat, c1.items_hat)

        c = ContentFiltering(num_users=users, num_items=items)
        test_helpers.assert_correct_num_users(users, c, c.users_hat.shape[0])
        test_helpers.assert_correct_num_items(items, c, c.items_hat.shape[1])
        test_helpers.assert_correct_size_generic(
            c.num_attributes, c.num_attributes, c.items_hat.shape[0]
        )
        test_helpers.assert_correct_size_generic(
            c.num_attributes, c.num_attributes, c.users_hat.shape[1]
        )
        test_helpers.assert_not_none(c.predicted_scores)

        # did not set seed, show random behavior
        c1 = ContentFiltering(num_users=users, num_items=items)

        with pytest.raises(AssertionError):
            test_helpers.assert_equal_arrays(c.items_hat, c1.items_hat)

        c = ContentFiltering(num_users=users, num_attributes=attr)
        test_helpers.assert_correct_num_users(users, c, c.users_hat.shape[0])
        test_helpers.assert_correct_num_items(c.num_items, c, c.items_hat.shape[1])
        test_helpers.assert_correct_size_generic(
            attr, c.num_attributes, c.items_hat.shape[0]
        )
        test_helpers.assert_correct_size_generic(
            attr, c.num_attributes, c.users_hat.shape[1]
        )
        test_helpers.assert_not_none(c.predicted_scores)

        # did not set seed, show random behavior
        c1 = ContentFiltering(num_users=users, num_attributes=attr)

        with pytest.raises(AssertionError):
            test_helpers.assert_equal_arrays(c.items_hat, c1.items_hat)

        c = ContentFiltering(num_attributes=attr, num_items=items)
        test_helpers.assert_correct_num_users(c.num_users, c, c.users_hat.shape[0])
        test_helpers.assert_correct_num_items(items, c, c.items_hat.shape[1])
        test_helpers.assert_correct_size_generic(
            attr, c.num_attributes, c.items_hat.shape[0]
        )
        test_helpers.assert_correct_size_generic(
            attr, c.num_attributes, c.users_hat.shape[1]
        )
        test_helpers.assert_not_none(c.predicted_scores)

        # did not set seed, show random behavior
        c1 = ContentFiltering(num_attributes=attr, num_items=items)

        with pytest.raises(AssertionError):
            test_helpers.assert_equal_arrays(c.items_hat, c1.items_hat)

    def test_representations(self, item_repr=None, user_repr=None, bad_user_repr=None):
        if item_repr is None:
            items = np.random.randint(5, 1000)
            attr = np.random.randint(5, 100)
            item_repr = np.random.random(size=(attr, items))
        if user_repr is None or user_repr.shape[1] != item_repr.shape[0]:
            users = np.random.randint(5, 100)
            user_repr = np.random.randint(10, size=(users, item_repr.shape[0]))

        c = ContentFiltering(item_representation=item_repr)
        test_helpers.assert_correct_num_users(c.num_users, c, c.users_hat.shape[0])
        test_helpers.assert_correct_num_items(
            item_repr.shape[1], c, c.items_hat.shape[1]
        )
        test_helpers.assert_correct_size_generic(
            item_repr.shape[0], c.num_attributes, c.items_hat.shape[0]
        )
        test_helpers.assert_correct_size_generic(
            item_repr.shape[0], c.num_attributes, c.users_hat.shape[1]
        )
        test_helpers.assert_equal_arrays(item_repr, c.items_hat)
        test_helpers.assert_not_none(c.predicted_scores)

        c = ContentFiltering(user_representation=user_repr)
        test_helpers.assert_correct_num_users(
            user_repr.shape[0], c, c.users_hat.shape[0]
        )
        test_helpers.assert_correct_num_items(c.num_items, c, c.items_hat.shape[1])
        test_helpers.assert_correct_size_generic(
            user_repr.shape[1], c.num_attributes, c.items_hat.shape[0]
        )
        test_helpers.assert_correct_size_generic(
            user_repr.shape[1], c.num_attributes, c.users_hat.shape[1]
        )
        test_helpers.assert_equal_arrays(user_repr, c.users_hat)
        test_helpers.assert_not_none(c.predicted_scores)

        # did not set seed, show random behavior
        c1 = ContentFiltering(user_representation=user_repr)

        with pytest.raises(AssertionError):
            # this assertion error might fail if items are randomly the same
            # also check attributes for users
            test_helpers.assert_equal_arrays(c.items_hat, c1.items_hat)

        c = ContentFiltering(
            user_representation=user_repr, item_representation=item_repr
        )
        test_helpers.assert_correct_num_users(
            user_repr.shape[0], c, c.users_hat.shape[0]
        )
        test_helpers.assert_correct_num_items(
            item_repr.shape[1], c, c.items_hat.shape[1]
        )
        test_helpers.assert_correct_size_generic(
            user_repr.shape[1], c.num_attributes, c.items_hat.shape[0]
        )
        test_helpers.assert_correct_size_generic(
            user_repr.shape[1], c.num_attributes, c.users_hat.shape[1]
        )
        test_helpers.assert_correct_size_generic(
            item_repr.shape[0], c.num_attributes, c.items_hat.shape[0]
        )
        test_helpers.assert_correct_size_generic(
            item_repr.shape[0], c.num_attributes, c.users_hat.shape[1]
        )
        test_helpers.assert_equal_arrays(user_repr, c.users_hat)
        test_helpers.assert_equal_arrays(item_repr, c.items_hat)
        test_helpers.assert_not_none(c.predicted_scores)

    def test_wrong_representation(
        self, user_repr=None, item_repr=None, bad_user_repr=None, bad_item_repr=None
    ):
        if item_repr is None:
            items = np.random.randint(1000)
            attr = np.random.randint(10)
            item_repr = np.random.random(size=(attr, items))
        if user_repr is None or user_repr.shape[1] != item_repr.shape[0]:
            users = np.random.randint(100)
            user_repr = np.random.randint(10, size=(users, item_repr.shape[0]))

        if bad_user_repr is None or bad_user_repr.shape[1] == item_repr.shape[0]:
            # |A| shouldn't match item_repr.shape[0]
            bad_user_repr = np.random.randint(
                10, size=(user_repr.shape[0], user_repr.shape[1] + 2)
            )
        if bad_item_repr is None or bad_item_repr.shape[0] == user_repr.shape[1]:
            # |A| shouldn't match user_repr.shape[1]
            bad_item_repr = np.random.random(
                size=(item_repr.shape[0] + 1, item_repr.shape[1])
            )

        with pytest.raises(ValueError):
            c = ContentFiltering(
                user_representation=bad_user_repr, item_representation=item_repr
            )
        with pytest.raises(ValueError):
            c = ContentFiltering(
                user_representation=user_repr, item_representation=bad_item_repr
            )
        with pytest.raises(ValueError):
            # actual user prefs and system's representation of user's prefs
            # must be the same dimension
            c = ContentFiltering(
                user_representation=user_repr,
                actual_user_scores=bad_user_repr,
                item_representation=bad_item_repr,
            )

    def test_additional_params(self, num_items_per_iter=None):
        if num_items_per_iter is None:
            num_items_per_iter = np.random.randint(5, 100)

        c = ContentFiltering(verbose=False, num_items_per_iter=num_items_per_iter)
        assert num_items_per_iter == c.num_items_per_iter
        # also check other params
        test_helpers.assert_correct_num_users(c.num_users, c, c.users_hat.shape[0])
        test_helpers.assert_correct_num_items(c.num_items, c, c.items_hat.shape[1])
        test_helpers.assert_correct_size_generic(
            c.num_attributes, c.num_attributes, c.users_hat.shape[1]
        )
        test_helpers.assert_correct_size_generic(
            c.num_attributes, c.num_attributes, c.items_hat.shape[0]
        )
        test_helpers.assert_not_none(c.predicted_scores)

    def test_seeding(self, seed=None, items=None, users=None):
        if seed is None:
            seed = np.random.randint(100000)
        s1 = ContentFiltering(seed=seed)
        s2 = ContentFiltering(seed=seed)
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
        s1 = ContentFiltering(seed=seed, num_users=users, num_items=items)
        s2 = ContentFiltering(seed=seed, num_users=users, num_items=items)
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

    def test_drift(self, seed=None, items=None, users=None):
        # user_repr:
        # [ [ 1 , 0 , ... , 0 ]
        #   [ 0 , 1 , ... , 0 ]
        #         ...
        #   [ 0 , 0 , ... , 1 ] ]
        user_repr = np.diag(np.ones(10))

        # item_repr (transposed):
        # [ [   1 , 0.1 ,   0 , ... , 0 ]
        #   [   0 ,   1 ,  0.1, ... , 0 ]
        #              ...
        #   [ 0.1 ,   0 ,   0 , ... , 1 ]
        # this is essentially the same as the user vector, with the addition
        # of a small value in an adjacent entry. we'll use this small value
        # to test whether users are correctly drifting towards the items
        # vector
        item_repr = (user_repr + 0.1 * np.vstack([user_repr[1:], user_repr[0]])).T
        users = Users(actual_user_profiles=np.copy(user_repr), num_users=10, drift=0.5)
        model = ContentFiltering(
            user_representation=np.copy(user_repr),
            item_representation=item_repr,
            actual_user_scores=users,
        )
        model.run(timesteps=1)
        # user profiles should have drifted after interacting with items
        assert not np.array_equal(user_repr, users.actual_user_profiles)
        # user profiles should be closer to the items after drifting
        orig_dist = np.linalg.norm(item_repr.T - user_repr, axis=1)
        new_dist = np.linalg.norm(item_repr.T - users.actual_user_profiles, axis=1)
        assert (new_dist < orig_dist).all()
        # let's go further and check that angles are decreased by 50% too!
        item_norm = normalize_matrix(item_repr.T)
        orig_angles = np.arccos((user_repr * item_norm).sum(axis=1))
        new_angles = np.arccos((users.actual_user_profiles * item_norm).sum(axis=1))
        np.testing.assert_array_almost_equal(0.5 * orig_angles, new_angles)
