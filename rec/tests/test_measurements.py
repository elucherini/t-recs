import test_utils
import numpy as np
from rec.models import SocialFiltering, ContentFiltering
from rec.metrics import Measurement, HomogeneityMeasurement, MSEMeasurement, DiffusionTreeMeasurement, StructuralVirality
import pytest

class MeasurementUtils:
    @classmethod
    def assert_valid_length(self, measurements, timesteps):
        # there are as many states as the timesteps for which we ran the system + the initial state
        for key, value in measurements.items():
            assert(len(value) == timesteps + 1)

    @classmethod
    def assert_valid_final_measurements(self, measurements, model_attribute,
                                        key_mappings, timesteps):
        for key, value in key_mappings.items():
            if key in measurements.keys():
                assert(np.array_equal(measurements[key][timesteps], value))
            else:
                assert(value not in s._system_state)

class TestMeasurementModule:
    """Test basic functionalities of MeasurementModule"""
    def test_measurement_module(self):
        # Create model, e.g., SocialFiltering
        s = SocialFiltering()
        # Add HomogeneityMeasurement
        old_metrics = s.metrics.copy()
        s.add_metrics(HomogeneityMeasurement())
        assert(len(old_metrics) + 1 == len(s.metrics))

        with pytest.raises(ValueError):
            s.add_metrics("wrong type")
        with pytest.raises(ValueError):
            s.add_metrics(MSEMeasurement(), print)
        with pytest.raises(ValueError):
            s.add_metrics()
        assert(len(old_metrics) + 1 == len(s.metrics))

    def test_system_state_module(self, timesteps=None):
        s = SocialFiltering()

        old_metrics = s._system_state.copy()

        with pytest.raises(ValueError):
            s.add_state_variable("wrong type")
        with pytest.raises(ValueError):
            s.add_state_variable(MSEMeasurement(), print)
        with pytest.raises(ValueError):
            s.add_state_variable()

    def test_run_and_measure(self, timesteps=None):
        if timesteps is None:
            timesteps = np.random.randint(2, 100)

        s = SocialFiltering()
        # TODO: this should be automated a bit
        state_mappings = {'predicted_user_profiles': s.user_profiles,
                        'actual_user_scores': s.actual_users.actual_user_scores,
                        'items': s.item_attributes,
                        'predicted_user_scores': s.predicted_scores}

        for t in range(1, timesteps + 1):
            s.run(timesteps=1)
            system_state = s.get_system_state()
            print(t)
            print(type(t))
            MeasurementUtils.assert_valid_final_measurements(system_state,
                                                             s._system_state,
                                                             state_mappings,
                                                             t)
            MeasurementUtils.assert_valid_length(system_state, t)

        s = SocialFiltering()

        for t in range(1, timesteps + 1):
            s.run(timesteps=1)
            measurements = s.get_measurements()
            MeasurementUtils.assert_valid_length(measurements, t)

class TestHomogeneityMeasurement():
    # TODO
    def test_generic(self):
        pass

class TestMSEMeasurement():
    # TODO
    pass
    """Test base class Measurement"""
    '''
    def test_generic(self, timesteps=None):
        # We do this through ContentFiltering
        c = ContentFiltering()
        c.add_measurements(MSEMeasurement())
        assert(len(c.metrics) > 0)

        # Run for some time
        if timesteps is None:
            timesteps = np.random.randint(100)
        c.run(timesteps=timesteps)
        meas = c.get_measurements()
        assert(meas is not None)
        assert(len(meas['MSE']) == timesteps + 1)
        # First element equal to NaN:
        assert(meas['MSE'][0] is None)
        # Non-increasing starting from second element
        print(meas['MSE'])
        #assert(all(x>=y for x, y in zip(meas['MSE'][1:], meas['MSE'][2:])))
    '''
