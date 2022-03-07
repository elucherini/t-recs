from trecs.models import ImplicitMF
from trecs.components import Users, Creators
import numpy as np
import scipy.sparse as sp
import pytest
import test_helpers


class TestImplicitMF:
    def test_default(self):
        mf = ImplicitMF(seed=123)
        test_helpers.assert_correct_num_users(mf.num_users, mf, mf.users_hat.num_users)
        test_helpers.assert_correct_num_items(mf.num_items, mf, mf.items_hat.num_items)
        # assert dimensions match up to latent features
        assert mf.num_latent_factors == mf.users_hat.num_attrs
        assert mf.num_latent_factors == mf.items_hat.num_attrs
        assert mf.num_latent_factors == 10
        assert mf.num_users == 100
        assert mf.num_items == 1250
        test_helpers.assert_not_none(mf.predicted_scores)

        # did not set seed, show random behavior
        mf2 = ImplicitMF(seed=1234)

        with pytest.raises(AssertionError):
            test_helpers.assert_equal_arrays(mf.items_hat, mf2.items_hat)

    def test_arguments(self):
        # assert defaults work properly
        mf = ImplicitMF()
        assert mf.num_users == 100
        assert mf.num_items == 1250
        assert mf.num_latent_factors == 10

        num_items = np.random.randint(1, 100)
        num_users = np.random.randint(1, 100)
        num_features = np.random.randint(1, 100)
        mf = ImplicitMF(num_users=num_users, num_items=num_items, num_latent_factors=num_features)
        assert mf.num_users == num_users
        assert mf.num_items == num_items
        assert mf.num_latent_factors == num_features

        with pytest.raises(ValueError):
            # pass in user and item representations that do not match number of
            # latent factors
            users_hat = np.random.random(size=(num_users, 101))
            items_hat = np.random.random(size=(101, num_items))
            mf = ImplicitMF(
                user_representation=users_hat,
                item_representation=items_hat,
                num_latent_factors=num_features,
            )

    def test_startup_run(self):
        mf = ImplicitMF(seed=123)
        items_hat_0, users_hat_0 = mf.items_hat.value.copy(), mf.users_hat.value.copy()
        num_startup_iters = 5
        mf.startup_and_train(num_startup_iters)
        # assert interactions are recorded
        assert mf.all_interactions.shape[0] == mf.num_users * num_startup_iters

        # the latent feature representation should have changed after fitting
        with pytest.raises(AssertionError):
            test_helpers.assert_equal_arrays(items_hat_0, mf.items_hat)
        with pytest.raises(AssertionError):
            test_helpers.assert_equal_arrays(users_hat_0, mf.users_hat)

        latent_items, latent_users = mf.items_hat.value.copy(), mf.users_hat.value.copy()
        mf.run(1)
        # should not have refit
        test_helpers.assert_equal_arrays(latent_items, mf.items_hat)
        test_helpers.assert_equal_arrays(latent_users, mf.users_hat)
        # total interactions should be num_users because there was one timestep;
        # interactions are reset at every call to run()
        assert mf.all_interactions.shape[0] == mf.num_users

        prior_scores = mf.predicted_scores.value.copy()
        mf.train()  # fit to interaction data from the most recent run
        with pytest.raises(AssertionError):
            # we should see new predicted scores
            test_helpers.assert_equal_arrays(prior_scores, mf.predicted_scores.value)
        with pytest.raises(AssertionError):
            # new item representation
            test_helpers.assert_equal_arrays(latent_items, mf.items_hat.value)
        with pytest.raises(AssertionError):
            # new user representation
            test_helpers.assert_equal_arrays(latent_users, mf.users_hat.value)

    def test_content_creators(self):
        # true users and true items
        num_users = 10
        num_creators = 5
        num_items = 20
        num_attrs = 5
        users = np.random.random(size=(num_users, num_attrs))
        items = np.random.random(size=(num_attrs, num_items))
        # 10 content creators
        creators = Creators(
            np.random.uniform(size=(num_creators, num_attrs)), creation_probability=1
        )
        num_factors = 3
        mf = ImplicitMF(
            num_latent_factors=num_factors,
            actual_user_representation=users,
            actual_item_representation=items,
            creators=creators,
        )
        # disallow content creators from making new items during startup phase
        mf.startup_and_train(5)
        # no new items should have been created during startup
        assert mf.items.num_items == num_items
        avg_item = mf.als_model.item_features_.T.mean(axis=1)
        mf.run(1)
        # there should be 5 new items with the same latent feature representation
        new_items_hat = mf.items_hat.value[:, -5:]
        test_helpers.assert_equal_arrays(np.tile(avg_item, (num_creators, 1)).T, new_items_hat)

    def test_sparse_matrix(self):
        num_users = 5
        num_items = 5
        users = sp.csr_matrix(np.eye(num_users))  # 5 users, 5 attributes
        items = sp.csr_matrix(np.eye(num_items))  # 5 users, 5 attributes

        model = ImplicitMF(
            actual_user_representation=users,
            actual_item_representation=items,
            num_items_per_iter=num_items,
        )
        init_pred_scores = model.predicted_user_item_scores.copy()
        # after one iteration of training, the model should have perfect
        # predictions, since each user was shown all the items in the item set
        model.startup_and_train(1)

        # assert new scores have changed
        trained_preds = model.predicted_user_item_scores.copy()
        with pytest.raises(AssertionError):
            test_helpers.assert_equal_arrays(init_pred_scores, trained_preds)

        # assert that recommendations are now "perfect"
        model.num_items_per_iter = 1
        recommendations = model.recommend()
        correct_rec = np.array([[0], [1], [2], [3], [4]])
        test_helpers.assert_equal_arrays(recommendations, correct_rec)

    def test_new_users(self):
        users = np.random.randint(10, size=(100, 10))
        items = np.random.randint(2, size=(10, 100))
        model = ImplicitMF(
            actual_user_representation=users,
            actual_item_representation=items,
        )
        model.run(1)
        num_new_users = 100
        users = np.random.randint(10, size=(num_new_users, 10))
        model.add_users(users)
        # 100 new users + 100 original = 200
        assert model.num_users == 200
        assert model.users.num_users == 200
        assert model.users_hat.num_users == 200
        assert model.users.actual_user_scores.num_users == 200
        # assert new users are represented as zeros
        user_representation = model.users_hat.value[100:, :].copy()
        assert model.all_interactions["user"].size == 100.0
        model.run(1, repeated_items=True, train_between_steps=True)
        # verify the user representation has changed after a new training step
        with pytest.raises(AssertionError):
            test_helpers.assert_equal_arrays(user_representation, model.users_hat.value[100:, :])
        # the first iteration should have yielded
        # 100 interactions, the second should yield another 200
        assert model.all_interactions["user"].size == 200.0
