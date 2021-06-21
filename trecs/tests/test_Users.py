import numpy as np
from trecs.components import Users, DNUsers
from trecs.models import ContentFiltering
import trecs.matrix_ops as mo
import test_helpers
import pytest


class TestUsers:
    def test_generic(self, num_items=10, num_attr=5, num_users=6, expand_items_by=2):
        with pytest.raises(ValueError):
            s = Users()
        with pytest.raises(TypeError):
            s = Users(actual_user_profiles="wrong type")
        with pytest.raises(TypeError):
            s = Users(actual_user_profiles=None, size="wrong type")
        with pytest.raises(TypeError):
            s = Users(size="wrong_type")
        s = Users(size=(num_users, num_attr))
        assert s.actual_user_profiles.shape == (num_users, num_attr)
        s = Users(actual_user_profiles=np.random.randint(5, size=(num_users, num_attr)))
        assert s.actual_user_profiles.shape == (num_users, num_attr)
        # can't normalize a vector that isn't a matrix
        s = Users(actual_user_profiles=np.array([[1, 2, 3]]))

    def test_content(self, num_items=10, num_attr=5, num_users=6):
        """WARNING Before running this, make sure ContentFiltering is working properly"""
        # user_repr = actual_user_repr
        item_repr = np.random.randint(2, size=(num_attr, num_items))
        actual_user_repr = np.random.randint(15, size=(num_users, num_attr))
        model = ContentFiltering(
            user_representation=actual_user_repr,
            item_representation=item_repr,
        )
        s = Users(actual_user_repr)
        s.set_score_function(model.score_fn)
        s.compute_user_scores(item_repr)
        model.train()
        test_helpers.assert_equal_arrays(s.actual_user_scores, model.predicted_scores)

        # user_repr != actual_user_repr
        user_repr = np.random.randint(15, size=(num_users, num_attr))
        model = ContentFiltering(user_representation=user_repr, item_representation=item_repr)
        assert model.users_hat.shape == actual_user_repr.shape
        s = Users(actual_user_repr)
        s.set_score_function(model.score_fn)
        s.compute_user_scores(item_repr)
        model.train()
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_array_equal(s.actual_user_scores, model.predicted_scores)

    def test_interact_with_items(self, num_users=15):
        def dummy_interact(users, items_shown):
            # return the first item shown
            return items_shown[:, 0]

        u = Users(interact_with_items=dummy_interact, size=(num_users, 1))
        items_shown = np.zeros((num_users, num_users))
        # the first item each user is shown corresponds to their own index
        items_shown[:, 0] = np.arange(num_users)
        result = u.get_user_feedback(items_shown=items_shown)
        np.testing.assert_array_equal(result, np.arange(num_users))

    def test_seeding(self, num_users=15, num_attr=15, seed=None):
        if seed is None:
            seed = np.random.randint(1000)
        users1 = Users(size=(num_users, num_attr), seed=seed)
        users2 = Users(size=(num_users, num_attr), seed=seed)
        test_helpers.assert_equal_arrays(users1.actual_user_profiles, users2.actual_user_profiles)
        # no seeding
        users3 = Users(size=(num_users, num_attr))
        users4 = Users(size=(num_users, num_attr))
        # very low chances of this passing
        with pytest.raises(AssertionError):
            test_helpers.assert_equal_arrays(
                users3.actual_user_profiles, users4.actual_user_profiles
            )

    def test_dn_users(self, num_users=15, num_attr=15, num_items=20, seed=None):
        users = DNUsers(size=(num_users, num_attr), seed=seed)
        items = np.random.uniform(size=(num_attr, num_items))
        user_item_scores = np.dot(users.actual_user_profiles.value, items)
        users.compute_user_scores(items)  # calculate underlying values

        # have a random sample of 5 num_items shown to each user
        items_shown = np.random.choice(num_items, size=(num_users, 5))
        shown_scores = user_item_scores[np.arange(num_users).reshape(-1, 1), items_shown]

        dn_utilities = users.calc_dn_utilities(shown_scores)
        # each of the 5 items should have an associated utility
        assert dn_utilities.shape == (num_users, 5)

        # randomly perturbed & normalized utilities should not be equal
        # to originally user-item scores
        with pytest.raises(AssertionError):
            test_helpers.assert_equal_arrays(shown_scores, dn_utilities)

    def test_value_normalization(self):
        users = DNUsers(np.array([[0, 1, 2, 3, 4]]), sigma=0.05, omega=0.25, beta=0.9)
        items = np.arange(20).reshape(5, 4)
        users.compute_user_scores(items)
        # should be: [120, 130, 140, 150]
        user_item_scores = np.dot(users.actual_user_profiles.value, items)
        normed_values = users.normalize_values(user_item_scores)

        # these are manually calculated
        normed_scores = np.array([[0.7620146], [0.82551582], [0.88901703], [0.95251825]])

        np.testing.assert_array_almost_equal(normed_values, normed_scores)

    def test_attention(self):
        num_users = 5
        num_attr = 4
        num_items = 5
        users = Users(np.ones((num_users, num_attr)), attention_exp=-0.8)
        items = np.ones((num_attr, num_items))
        # item at index 1 has a slightly higher score, without the attentional
        # mechanism, all users would score it the highest
        items[:, 1] = 1.5
        users.compute_user_scores(items)
        items_shown = np.tile(np.arange(num_items), (num_users, 1))
        feedback = users.get_user_feedback(items_shown)
        np.testing.assert_array_equal(feedback, np.zeros(num_users))

        # when we turn user attention off, we should see that all users
        # interact with item 1
        users.attention_exp = 0
        users.compute_user_scores(items)
        feedback = users.get_user_feedback(items_shown)
        np.testing.assert_array_equal(feedback, np.ones(num_users))

    def test_no_repeated_items(self):
        num_users = 5
        num_attr = 4
        num_items = 5
        users = Users(np.ones((num_users, num_attr)), repeat_interactions=False)
        # first item is most desirable because its attributes are all 0,
        # second item is second most desirable because attributes are all -1, etc.
        items = np.tile(np.arange(num_items) * -1, (num_attr, 1))
        users.compute_user_scores(items)
        items_shown = np.tile(np.arange(num_items), (num_users, 1))  # all items shown to all users
        # all users should like the 0th item best
        feedback = users.get_user_feedback(items_shown)
        np.testing.assert_array_equal(feedback, np.zeros(num_users))
        # all users should then like the 1st item, since they're not
        # allowed to repeat interactions
        feedback = users.get_user_feedback(items_shown)
        np.testing.assert_array_equal(feedback, np.ones(num_users))
        # all users should then like the 2nd item, since they're not
        # allowed to repeat interactions
        feedback = users.get_user_feedback(items_shown)
        np.testing.assert_array_equal(feedback, np.ones(num_users) * 2)
        # test that all users still have well-defined score values;
        # i.e., no score values are equal to negative infinity, they
        # should be equal to the orignial score values
        correct_scores = mo.inner_product(users.actual_user_profiles.value, items)
        np.testing.assert_array_equal(users.actual_user_scores.value, correct_scores)
