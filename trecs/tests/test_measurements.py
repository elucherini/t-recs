import test_helpers
import numpy as np
import pytest
from trecs.components import Users, Items
from trecs.models import SocialFiltering, ContentFiltering, PopularityRecommender, BassModel
from trecs.metrics import (
    InteractionSpread,
    MSEMeasurement,
    DiffusionTreeMeasurement,
    StructuralVirality,
    InteractionMeasurement,
    RecSimilarity,
    InteractionSimilarity,
    AverageFeatureScoreRange,
    RMSEMeasurement,
)


class MeasurementUtils:
    @classmethod
    def assert_valid_length(self, measurements, timesteps):
        # there are as many states as the timesteps for which we ran the
        # system, plus one to account for the initial state
        for _, value in measurements.items():
            assert len(value) == timesteps + 1

    @classmethod
    def assert_valid_final_measurements(
        self, measurements, model_attribute, key_mappings, timesteps
    ):
        for key, value in key_mappings.items():
            if key in measurements.keys():
                assert np.array_equal(measurements[key][timesteps], value)
            else:
                assert value not in model_attribute

    @classmethod
    def test_generic_metric(self, model, metric, timesteps):
        if metric not in model.metrics:
            model.add_metrics(metric)
        assert metric in model.metrics

        for t in range(1, timesteps + 1):
            model.run(timesteps=1)
            measurements = model.get_measurements()
            self.assert_valid_length(measurements, t)


class TestMeasurementModule:
    """Test basic functionalities of MeasurementModule"""

    def test_measurement_module(self):
        # Create model, e.g., SocialFiltering
        s = SocialFiltering()
        # Add Interaction Spread
        old_metrics = s.metrics.copy()
        s.add_metrics(InteractionSpread())
        assert len(old_metrics) + 1 == len(s.metrics)

        with pytest.raises(ValueError):
            s.add_metrics("wrong type")
        with pytest.raises(ValueError):
            s.add_metrics(MSEMeasurement(), print)
        with pytest.raises(ValueError):
            s.add_metrics()
        assert len(old_metrics) + 1 == len(s.metrics)

    def test_system_state_module(self):
        s = SocialFiltering()

        with pytest.raises(ValueError):
            s.add_state_variable("wrong type")
        with pytest.raises(ValueError):
            s.add_state_variable(MSEMeasurement(), print)
        with pytest.raises(ValueError):
            s.add_state_variable()

    def test_default_measurements(self, timesteps=None):
        if timesteps is None:
            timesteps = np.random.randint(2, 100)

        s = SocialFiltering(record_base_state=True)

        for t in range(1, timesteps + 1):
            s.run(timesteps=1)
            system_state = s.get_system_state()
            state_mappings = {
                "predicted_users": s.users_hat.value,
                "actual_user_scores": s.users.actual_user_scores.value,
                "predicted_items": s.items_hat.value,
                "predicted_user_scores": s.predicted_scores.value,
            }
            MeasurementUtils.assert_valid_final_measurements(
                system_state, s._system_state, state_mappings, t
            )
            MeasurementUtils.assert_valid_length(system_state, t)

        s = SocialFiltering()
        s.add_metrics(MSEMeasurement())

        for t in range(1, timesteps + 1):
            s.run(timesteps=1)
            measurements = s.get_measurements()
            MeasurementUtils.assert_valid_length(measurements, t)


class TestInteractionSpread:
    def test_generic(self, timesteps=None):
        if timesteps is None:
            timesteps = np.random.randint(2, 100)
        MeasurementUtils.test_generic_metric(SocialFiltering(), InteractionSpread(), timesteps)
        MeasurementUtils.test_generic_metric(
            PopularityRecommender(), InteractionSpread(), timesteps
        )


class TestRecSimilarity:
    def test_generic(self, timesteps=None):
        if timesteps is None:
            timesteps = np.random.randint(2, 100)
        # default # of users is 100
        pairs = [np.random.choice(100, 2, replace=False) for i in range(50)]
        MeasurementUtils.test_generic_metric(SocialFiltering(), RecSimilarity(pairs), timesteps)
        MeasurementUtils.test_generic_metric(
            PopularityRecommender(), RecSimilarity(pairs), timesteps
        )


class TestInteractionSimilarity:
    def test_generic(self, timesteps=None):
        if timesteps is None:
            timesteps = np.random.randint(2, 100)
        # default # of users is 100
        pairs = [np.random.choice(100, 2, replace=False) for i in range(50)]
        MeasurementUtils.test_generic_metric(
            SocialFiltering(), InteractionSimilarity(pairs), timesteps
        )
        MeasurementUtils.test_generic_metric(
            PopularityRecommender(), RecSimilarity(pairs), timesteps
        )

    def test_functionality(self):
        num_users = 2
        users = np.eye(num_users)
        items = np.zeros((2, 1))
        # only one item so the jaccard similarity should be 1
        pairs = [(0, 1)]
        content = ContentFiltering(
            actual_user_representation=users, actual_item_representation=items, num_items_per_iter=1
        )
        content.add_metrics(InteractionSimilarity(pairs))
        content.run(5)
        final_jacc = content.get_measurements()["interaction_similarity"][-1]
        assert final_jacc == 1  # both users have the item interaction index

        new_items = np.eye(num_users)
        content = ContentFiltering(
            user_representation=users,
            item_representation=new_items,
            actual_user_representation=users,
            actual_item_representation=new_items,
            num_items_per_iter=2,
        )
        content.add_metrics(InteractionSimilarity(pairs))
        content.run(5)
        final_jacc = content.get_measurements()["interaction_similarity"][-1]
        assert final_jacc == 0  # users have no interactions in common

        # alter items such that both users prefer the first item
        new_items[:, 0] = np.ones(2)
        new_items[:, 1] = np.zeros(2)
        content.items = Items(new_items)
        # force users to recalculate scores
        content.users.compute_user_scores(new_items)
        content.run(1)
        final_jacc = content.get_measurements()["interaction_similarity"][-1]
        assert final_jacc == 0.5  # users should now have 1 interaction item in common


class TestMSEMeasurement:
    def test_generic(self, timesteps=None):
        if timesteps is None:
            timesteps = np.random.randint(2, 100)
        MeasurementUtils.test_generic_metric(SocialFiltering(), MSEMeasurement(), timesteps)
        MeasurementUtils.test_generic_metric(PopularityRecommender(), MSEMeasurement(), timesteps)

    def test_numeric(self):
        pop = PopularityRecommender()
        pop.add_metrics(MSEMeasurement())
        pop.run(5)
        mse = np.array(pop.get_measurements()["mse"][1:])
        assert not np.isnan(mse).any()
        assert not np.isinf(mse).any()


class TestRMSEMeasurement:
    def test_generic(self, timesteps=None):
        if timesteps is None:
            timesteps = np.random.randint(2, 100)
        MeasurementUtils.test_generic_metric(SocialFiltering(), RMSEMeasurement(), timesteps)
        MeasurementUtils.test_generic_metric(PopularityRecommender(), RMSEMeasurement(), timesteps)

    def test_functionality(self):
        num_users, num_attrs, num_items = 100, 20, 100
        user_profiles = np.random.randint(2, size=(num_users, num_attrs))
        items = np.random.randint(2, size=(num_attrs, num_items))
        users = Users(actual_user_profiles=user_profiles, repeat_interactions=False)
        pop = PopularityRecommender(
            actual_user_representation=users, actual_item_representation=items
        )
        pop.add_metrics(MSEMeasurement(), RMSEMeasurement())
        pop.run(5)
        mse = np.array(pop.get_measurements()["mse"][1:])
        rmse = np.array(pop.get_measurements()["rmse"][1:])
        np.testing.assert_almost_equal(np.sqrt(mse), rmse)
        assert not np.isnan(rmse).any()
        assert not np.isinf(rmse).any()


class TestAFSRMeasurement:
    def test_generic(self, timesteps=None):
        if timesteps is None:
            timesteps = np.random.randint(2, 100)
        # not intended for binary features, so generate item representation
        item_representation = np.random.random((20, 100))
        MeasurementUtils.test_generic_metric(
            SocialFiltering(item_representation=item_representation),
            AverageFeatureScoreRange(),
            timesteps,
        )
        MeasurementUtils.test_generic_metric(
            ContentFiltering(item_representation=item_representation),
            AverageFeatureScoreRange(),
            timesteps,
        )


class TestInteractionMeasurement:
    def test_generic(self, timesteps=None):
        if timesteps is None:
            timesteps = np.random.randint(2, 100)
        MeasurementUtils.test_generic_metric(SocialFiltering(), InteractionMeasurement(), timesteps)
        MeasurementUtils.test_generic_metric(
            PopularityRecommender(), InteractionMeasurement(), timesteps
        )

    def test_interactions(self, timesteps=None):
        if timesteps is None:
            timesteps = np.random.randint(2, 100)

        s = SocialFiltering()
        s.add_metrics(InteractionMeasurement())

        for t in range(1, timesteps + 1):
            s.run(timesteps=1)
            measurements = s.get_measurements()
            histogram = measurements[InteractionMeasurement().name]
            if t > 1:
                # Check that there's one interaction per user
                assert np.array_equal(histogram[-1], histogram[t])
                assert histogram[-1].sum() == s.num_users


class TestDiffusionTreeMeasurement:
    def test_generic(self, timesteps=None):
        if timesteps is None:
            timesteps = np.random.randint(2, 100)
        b = BassModel()
        MeasurementUtils.test_generic_metric(
            b, DiffusionTreeMeasurement(b.infection_state), timesteps
        )


class TestStructuralVirality:
    def test_generic(self, timesteps=None):
        if timesteps is None:
            timesteps = np.random.randint(2, 100)
        b = BassModel()
        MeasurementUtils.test_generic_metric(b, StructuralVirality(b.infection_state), timesteps)
