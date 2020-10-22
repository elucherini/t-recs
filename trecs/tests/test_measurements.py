import test_helpers
import numpy as np
from trecs.models import SocialFiltering, ContentFiltering, BassModel
from trecs.metrics import (
    Measurement,
    HomogeneityMeasurement,
    MSEMeasurement,
    DiffusionTreeMeasurement,
    StructuralVirality,
    InteractionMeasurement,
    JaccardSimilarity,
)
import pytest


class MeasurementUtils:
    @classmethod
    def assert_valid_length(self, measurements, timesteps):
        # there are as many states as the timesteps for which we ran the
        # system + the initial state
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
                assert value not in s._system_state

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
        # Add HomogeneityMeasurement
        old_metrics = s.metrics.copy()
        s.add_metrics(HomogeneityMeasurement())
        assert len(old_metrics) + 1 == len(s.metrics)

        with pytest.raises(ValueError):
            s.add_metrics("wrong type")
        with pytest.raises(ValueError):
            s.add_metrics(MSEMeasurement(), print)
        with pytest.raises(ValueError):
            s.add_metrics()
        assert len(old_metrics) + 1 == len(s.metrics)

    def test_system_state_module(self, timesteps=None):
        s = SocialFiltering()

        old_metrics = s._system_state.copy()

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
        state_mappings = {
            "predicted_user_profiles": s.users_hat,
            "actual_user_scores": s.users.actual_user_scores,
            "items": s.items_hat,
            "predicted_user_scores": s.predicted_scores,
        }

        for t in range(1, timesteps + 1):
            s.run(timesteps=1)
            system_state = s.get_system_state()
            MeasurementUtils.assert_valid_final_measurements(
                system_state, s._system_state, state_mappings, t
            )
            MeasurementUtils.assert_valid_length(system_state, t)

        s = SocialFiltering()

        for t in range(1, timesteps + 1):
            s.run(timesteps=1)
            measurements = s.get_measurements()
            MeasurementUtils.assert_valid_length(measurements, t)


class TestHomogeneityMeasurement:
    def test_generic(self, timesteps=None):
        if timesteps is None:
            timesteps = np.random.randint(2, 100)
        MeasurementUtils.test_generic_metric(SocialFiltering(), HomogeneityMeasurement(), timesteps)
        MeasurementUtils.test_generic_metric(
            ContentFiltering(), HomogeneityMeasurement(), timesteps
        )


class TestJaccardSimilarity:
    def test_generic(self, timesteps=None):
        if timesteps is None:
            timesteps = np.random.randint(2, 100)
        # default # of users is 100
        pairs = [np.random.choice(100, 2, replace=False) for i in range(50)]
        MeasurementUtils.test_generic_metric(SocialFiltering(), JaccardSimilarity(pairs), timesteps)
        MeasurementUtils.test_generic_metric(
            ContentFiltering(), JaccardSimilarity(pairs), timesteps
        )


class TestMSEMeasurement:
    def test_generic(self, timesteps=None):
        if timesteps is None:
            timesteps = np.random.randint(2, 100)
        MeasurementUtils.test_generic_metric(SocialFiltering(), MSEMeasurement(), timesteps)
        MeasurementUtils.test_generic_metric(ContentFiltering(), MSEMeasurement(), timesteps)


class TestInteractionMeasurement:
    def test_generic(self, timesteps=None):
        if timesteps is None:
            timesteps = np.random.randint(2, 100)
        MeasurementUtils.test_generic_metric(SocialFiltering(), InteractionMeasurement(), timesteps)
        MeasurementUtils.test_generic_metric(
            ContentFiltering(), InteractionMeasurement(), timesteps
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
