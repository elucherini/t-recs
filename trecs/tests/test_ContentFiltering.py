from attr import attrs
import numpy as np
import scipy.sparse as sp
import pytest
from trecs.metrics.measurement import MSEMeasurement
from trecs.models import ContentFiltering
from trecs.components import Users, Creators
import test_helpers
import trecs.matrix_ops as mo


class TestContentFiltering:
    def test_default(self):
        c = ContentFiltering()
        test_helpers.assert_correct_num_users(c.num_users, c, c.users_hat.num_users)
        test_helpers.assert_correct_num_items(c.num_items, c, c.items_hat.num_items)
        # test attributes
        assert c.users_hat.num_attrs == c.items_hat.num_attrs
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
        test_helpers.assert_correct_num_users(users, c, c.users_hat.num_users)
        test_helpers.assert_correct_num_items(items, c, c.items_hat.num_items)
        # assert attributes look correct
        assert attr == c.items_hat.num_attrs
        assert attr == c.users_hat.num_attrs
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
        test_helpers.assert_correct_num_users(users, c, c.users_hat.num_users)
        test_helpers.assert_correct_num_items(c.num_items, c, c.items_hat.num_items)
        # assert attributes look correct
        assert c.users_hat.num_attrs == c.items_hat.num_attrs
        test_helpers.assert_not_none(c.predicted_scores)

        # did not set seed, show random behavior
        c1 = ContentFiltering(num_users=users)

        with pytest.raises(AssertionError):
            test_helpers.assert_equal_arrays(c.items_hat, c1.items_hat)

        c = ContentFiltering(num_items=items)
        test_helpers.assert_correct_num_users(c.num_users, c, c.users_hat.num_users)
        test_helpers.assert_correct_num_items(items, c, c.items_hat.num_items)
        # assert attributes look correct
        assert c.users_hat.num_attrs == c.items_hat.num_attrs
        test_helpers.assert_not_none(c.predicted_scores)

        # did not set seed, show random behavior
        c1 = ContentFiltering(num_items=items)

        with pytest.raises(AssertionError):
            test_helpers.assert_equal_arrays(c.items_hat, c1.items_hat)

        c = ContentFiltering(num_attributes=attr)
        test_helpers.assert_correct_num_users(c.num_users, c, c.users_hat.num_users)
        test_helpers.assert_correct_num_items(c.num_items, c, c.items_hat.num_items)
        # assert attributes look correct
        assert attr == c.users_hat.num_attrs
        assert attr == c.items_hat.num_attrs
        test_helpers.assert_not_none(c.predicted_scores)

        # did not set seed, show random behavior
        c1 = ContentFiltering(num_attributes=attr)

        with pytest.raises(AssertionError):
            test_helpers.assert_equal_arrays(c.items_hat, c1.items_hat)

        c = ContentFiltering(num_users=users, num_items=items)
        test_helpers.assert_correct_num_users(users, c, c.users_hat.num_users)
        test_helpers.assert_correct_num_items(items, c, c.items_hat.num_items)
        # assert attributes look correct
        assert c.users_hat.num_attrs == c.items_hat.num_attrs
        test_helpers.assert_not_none(c.predicted_scores)

        # did not set seed, show random behavior
        c1 = ContentFiltering(num_users=users, num_items=items)

        with pytest.raises(AssertionError):
            test_helpers.assert_equal_arrays(c.items_hat, c1.items_hat)

        c = ContentFiltering(num_users=users, num_attributes=attr)
        test_helpers.assert_correct_num_users(users, c, c.users_hat.num_users)
        test_helpers.assert_correct_num_items(c.num_items, c, c.items_hat.num_items)
        # assert attributes look correct
        assert attr == c.users_hat.num_attrs
        assert attr == c.items_hat.num_attrs
        test_helpers.assert_not_none(c.predicted_scores)

        # did not set seed, show random behavior
        c1 = ContentFiltering(num_users=users, num_attributes=attr)

        with pytest.raises(AssertionError):
            test_helpers.assert_equal_arrays(c.items_hat, c1.items_hat)

        c = ContentFiltering(num_attributes=attr, num_items=items)
        test_helpers.assert_correct_num_users(c.num_users, c, c.users_hat.num_users)
        test_helpers.assert_correct_num_items(items, c, c.items_hat.num_items)
        # assert attributes look correct
        assert attr == c.users_hat.num_attrs
        assert attr == c.items_hat.num_attrs
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

        c = ContentFiltering(item_representation=item_repr, seed=101)
        test_helpers.assert_correct_num_users(c.num_users, c, c.users_hat.num_users)
        test_helpers.assert_correct_num_items(item_repr.shape[1], c, c.items_hat.num_items)
        test_helpers.assert_correct_size_generic(item_repr.shape[0], attr, c.items_hat.num_attrs)
        test_helpers.assert_correct_size_generic(item_repr.shape[0], attr, c.users_hat.num_attrs)
        test_helpers.assert_equal_arrays(item_repr, c.items_hat)
        test_helpers.assert_not_none(c.predicted_scores)

        c = ContentFiltering(user_representation=user_repr, seed=102)
        test_helpers.assert_correct_num_users(user_repr.shape[0], c, c.users_hat.num_users)
        test_helpers.assert_correct_num_items(c.num_items, c, c.items_hat.num_items)
        test_helpers.assert_correct_size_generic(user_repr.shape[1], attr, c.items_hat.num_attrs)
        test_helpers.assert_correct_size_generic(user_repr.shape[1], attr, c.users_hat.num_attrs)
        test_helpers.assert_equal_arrays(user_repr, c.users_hat)
        test_helpers.assert_not_none(c.predicted_scores)

        # did not set seed, show random behavior
        c1 = ContentFiltering(user_representation=user_repr)

        with pytest.raises(AssertionError):
            # this assertion error might fail if items are randomly the same
            # also check attributes for users
            test_helpers.assert_equal_arrays(c.items_hat, c1.items_hat)

        c = ContentFiltering(user_representation=user_repr, item_representation=item_repr)
        test_helpers.assert_correct_num_users(user_repr.shape[0], c, c.users_hat.num_users)
        test_helpers.assert_correct_num_items(item_repr.shape[1], c, c.items_hat.num_items)
        test_helpers.assert_correct_size_generic(user_repr.shape[1], attr, c.items_hat.num_attrs)
        test_helpers.assert_correct_size_generic(user_repr.shape[1], attr, c.users_hat.num_attrs)
        test_helpers.assert_correct_size_generic(item_repr.shape[0], attr, c.items_hat.num_attrs)
        test_helpers.assert_correct_size_generic(item_repr.shape[0], attr, c.users_hat.num_attrs)
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
            bad_user_repr = np.random.randint(10, size=(user_repr.shape[0], user_repr.shape[1] + 2))
        if bad_item_repr is None or bad_item_repr.shape[0] == user_repr.shape[1]:
            # |A| shouldn't match user_repr.shape[1]
            bad_item_repr = np.random.random(size=(item_repr.shape[0] + 1, item_repr.shape[1]))

        with pytest.raises(ValueError):
            ContentFiltering(user_representation=bad_user_repr, item_representation=item_repr)
        with pytest.raises(ValueError):
            ContentFiltering(user_representation=user_repr, item_representation=bad_item_repr)
        with pytest.raises(ValueError):
            # actual user prefs and system's representation of user's prefs
            # must be the same dimension
            ContentFiltering(
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
        test_helpers.assert_correct_num_users(c.num_users, c, c.users_hat.num_users)
        test_helpers.assert_correct_num_items(c.num_items, c, c.items_hat.num_items)
        assert c.users_hat.num_attrs == c.items_hat.num_attrs
        test_helpers.assert_not_none(c.predicted_scores)

    def test_seeding(self, seed=None, items=None, users=None):
        if seed is None:
            seed = np.random.randint(100000)
        s1 = ContentFiltering(seed=seed, record_base_state=True)
        s1.add_metrics(MSEMeasurement())
        s2 = ContentFiltering(seed=seed, record_base_state=True)
        s2.add_metrics(MSEMeasurement())
        test_helpers.assert_equal_arrays(s1.items_hat, s2.items_hat)
        test_helpers.assert_equal_arrays(s1.users_hat, s2.users_hat)
        s1.run(timesteps=2)
        s2.run(timesteps=2)
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
        s1 = ContentFiltering(seed=seed, num_users=users, num_items=items, record_base_state=True)
        s1.add_metrics(MSEMeasurement())
        s2 = ContentFiltering(seed=seed, num_users=users, num_items=items, record_base_state=True)
        s2.add_metrics(MSEMeasurement())
        test_helpers.assert_equal_arrays(s1.items_hat, s2.items_hat)
        test_helpers.assert_equal_arrays(s1.users_hat, s2.users_hat)
        s1.run(timesteps=2)
        s2.run(timesteps=2)
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

        model = ContentFiltering(
            actual_user_representation=users,
            actual_item_representation=items,
            num_items_per_iter=num_items,
        )
        init_pred_scores = model.predicted_user_item_scores.copy()
        # after one iteration of training, the model should have perfect
        # predictions, since each user was shown all the items in the item set
        model.run(1)

        # assert new scores have changed
        trained_preds = model.predicted_user_item_scores.copy()
        with pytest.raises(AssertionError):
            test_helpers.assert_equal_arrays(init_pred_scores, trained_preds)

        # assert that recommendations are now "perfect"
        model.num_items_per_iter = 1
        recommendations = model.recommend()
        correct_rec = np.array([[0], [1], [2], [3], [4]])
        test_helpers.assert_equal_arrays(recommendations, correct_rec)

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
        users = Users(actual_user_profiles=user_repr.copy(), num_users=10, drift=0.5)
        model = ContentFiltering(
            user_representation=user_repr.copy(),
            item_representation=item_repr,
            actual_user_representation=users,
        )
        model.run(timesteps=1)
        # user profiles should have drifted after interacting with items
        assert not np.array_equal(user_repr, users.actual_user_profiles)
        # user profiles should be closer to the items after drifting
        orig_dist = np.linalg.norm(item_repr.T - user_repr, axis=1)
        new_dist = np.linalg.norm(item_repr.T - users.actual_user_profiles.value, axis=1)
        assert (new_dist < orig_dist).all()
        # let's go further and check that angles are decreased by 50% too!
        item_norm = mo.normalize_matrix(item_repr.T)
        orig_angles = np.arccos((user_repr * item_norm).sum(axis=1))
        new_angles = np.arccos((users.actual_user_profiles.value * item_norm).sum(axis=1))
        np.testing.assert_array_almost_equal(0.5 * orig_angles, new_angles)

    def test_creator_items(self):
        users = np.random.randint(10, size=(100, 10))
        items = np.random.randint(2, size=(10, 100))
        creator_profiles = Creators(
            np.random.uniform(size=(50, 10)), creation_probability=1.0
        )  # 50 creator profiles
        cf = ContentFiltering(
            actual_user_representation=users, item_representation=items, creators=creator_profiles
        )
        cf.run(1, repeated_items=True)
        assert cf.items.value.shape == (10, 150)  # 50 new items
        assert cf.items_hat.value.shape == (10, 150)
        assert cf.users.actual_user_scores.state_history[-1].shape == (100, 150)

    def test_sparse_matrix(self):
        num_users = 5
        num_items = 5
        num_attrs = 10
        users = sp.csr_matrix(np.eye(num_users))  # 5 users, 5 attributes
        items = sp.csr_matrix(np.eye(num_items))  # 5 users, 5 attributes
        users_hat = sp.csr_matrix((num_users, num_attrs))
        items_hat = sp.csr_matrix(
            mo.normalize_matrix(np.random.random((num_attrs, num_items)), axis=0)
        )

        model = ContentFiltering(
            user_representation=users_hat.copy(),
            item_representation=items_hat.copy(),
            actual_user_representation=users.copy(),
            actual_item_representation=items.copy(),
            num_items_per_iter=num_items,
        )
        init_pred_scores = mo.to_dense(model.predicted_user_item_scores.copy())
        # after one iteration of training, the model should have perfect
        # predictions, since each user was shown all the items in the item set
        model.run(1)

        # assert new scores have changed
        trained_preds = mo.to_dense(model.predicted_user_item_scores.copy())
        with pytest.raises(AssertionError):
            test_helpers.assert_equal_arrays(init_pred_scores, trained_preds)

        # assert that recommendations are now "perfect"
        model.num_items_per_iter = 1
        recommendations = model.recommend()
        correct_rec = np.array([[0], [1], [2], [3], [4]])
        test_helpers.assert_equal_arrays(recommendations, correct_rec)

        # ensure no errors when we pass in different sparse matrices
        model = ContentFiltering(
            user_representation=users_hat.copy(),
            item_representation=items_hat.copy(),
            num_items_per_iter=num_items,
        )
        model.run(1)

        model = ContentFiltering(
            actual_user_representation=users.copy(),
            actual_item_representation=items.copy(),
            num_items_per_iter=num_items,
        )
        model.run(1)

    def test_new_users(self):
        users = np.random.randint(10, size=(100, 10))
        items = np.random.randint(2, size=(10, 100))
        model = ContentFiltering(
            actual_user_representation=users,
            actual_item_representation=items,
        )
        model.run(1, repeated_items=True)
        num_new_users = 100
        users = np.random.randint(10, size=(num_new_users, 10))
        model.add_users(users)
        # 100 new users + 100 original = 200
        assert model.num_users == 200
        assert model.users.num_users == 200
        assert model.users_hat.num_users == 200
        assert model.users.actual_user_scores.num_users == 200
        # assert new users are represented as zeros
        user_representation = np.zeros((100, model.users_hat.num_attrs))
        test_helpers.assert_equal_arrays(user_representation, model.users_hat.value[100:, :])
        assert model.all_interactions.sum() == 100.0
        model.run(1, repeated_items=True)
        # verify the user representation has changed after a new training step
        with pytest.raises(AssertionError):
            test_helpers.assert_equal_arrays(user_representation, model.users_hat.value[100:, :])
        # the first iteration should have yielded
        # 100 interactions, the second should yield another 200
        assert model.all_interactions.sum() == 300.0
