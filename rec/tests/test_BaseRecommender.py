import test_utils
import numpy as np
from rec.models import SocialFiltering
from rec.metrics import HomogeneityMeasurement, MSEMeasurement
import pytest

class TestBaseRecommender:
    """Test basic functionalities of BaseRecommender"""
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

