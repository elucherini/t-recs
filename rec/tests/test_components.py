from rec.models.recommender import SystemStateModule
from rec.components import PredictedUserProfiles
import numpy as np


class TestComponents:
    def test_system_state(self):
        profiles = PredictedUserProfiles(np.zeros((5, 5)))
        sys = SystemStateModule()
        sys.add_state_variable(profiles)

        # increment profile twice
        for _ in range(2):
            profiles += 1
            for component in sys._system_state:
                component.store_state()

        assert len(profiles.state_history) == 3
        np.testing.assert_array_equal(profiles.state_history[0], np.zeros((5, 5)))
        np.testing.assert_array_equal(profiles.state_history[1], 1 * np.ones((5, 5)))
        np.testing.assert_array_equal(profiles.state_history[2], 2 * np.ones((5, 5)))
