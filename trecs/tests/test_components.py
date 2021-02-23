from trecs.models.recommender import SystemStateModule
from trecs.components import PredictedUserProfiles
import numpy as np


class TestComponents:
    def test_system_state(self):
        profiles = PredictedUserProfiles(np.zeros((5, 5)))
        sys = SystemStateModule()
        sys.add_state_variable(profiles)

        # increment profile twice
        for _ in range(2):
            profiles.value += 1
            for component in sys._system_state:
                component.store_state()

        assert len(profiles.state_history) == 3
        np.testing.assert_array_equal(profiles.state_history[0], np.zeros((5, 5)))
        np.testing.assert_array_equal(profiles.state_history[1], 1 * np.ones((5, 5)))
        np.testing.assert_array_equal(profiles.state_history[2], 2 * np.ones((5, 5)))

    def test_closed_logger(self):
        profiles = PredictedUserProfiles(np.zeros((5, 5)))
        logger = profiles._logger.logger  # pylint: disable=protected-access
        handler = profiles._logger.handler  # pylint: disable=protected-access
        assert len(logger.handlers) > 0  # before garbage collection
        del profiles
        # after garbage collection, handler should be closed
        assert handler not in logger.handlers
