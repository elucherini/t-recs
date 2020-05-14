import test_utils
import numpy as np
from rec.models import SocialFiltering, ContentFiltering
from rec.metrics import Measurement, HomogeneityMeasurement, MSEMeasurement, DiffusionTreeMeasurement, StructuralVirality
import pytest

class TestMeasurementModule:
    """Test basic functionalities of MeasurementModule"""
    def test_measurement_module(self):
        # Create model, e.g., SocialFiltering
        s = SocialFiltering()
        # Add HomogeneityMeasurement
        old_metrics = s.measurements.copy()
        s.add_measurements(HomogeneityMeasurement())
        assert(len(old_metrics) + 1 == len(s.measurements))

        with pytest.raises(ValueError):
            s.add_measurements("wrong type")
        with pytest.raises(ValueError):
            s.add_measurements(MSEMeasurement(), print)
        with pytest.raises(ValueError):
            s.add_measurements()
        assert(len(old_metrics) + 1 == len(s.measurements))

class TestMSEMeasurement:
    """Test base class Measurement"""
    def test_generic(self, timesteps=None):
        # We do this through SocialFiltering again
        c = ContentFiltering()
        c.add_measurements(MSEMeasurement())
        assert(len(c.measurements) > 0)

        # Run for some time
        if timesteps is None:
            timesteps = np.random.randint(100)
        c.run(timesteps=timesteps)
        meas = c.get_measurements()
        assert(meas is not None)
        assert(len(meas['MSE']) == timesteps + 1)
        # First element equal to NaN:
        assert(meas['MSE'][0] is None)
        # Non-decreasing starting from second element
        assert(all(x>=y for x, y in zip(meas['MSE'][1:], meas['MSE'][2:])))
