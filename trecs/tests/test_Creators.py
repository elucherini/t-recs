import numpy as np
from trecs.components import Creators
from trecs.models import ContentFiltering
import test_helpers
import pytest


class TestCreators:
    def test_generic(self, items=10, attr=5, creators=6, expand_items_by=2):
        with pytest.raises(ValueError):
            c = Creators()
        with pytest.raises(TypeError):
            c = Creators(actual_creator_profiles="wrong type")
        with pytest.raises(TypeError):
            c = Creators(actual_creator_profiles=None, size="wrong type")
        with pytest.raises(TypeError):
            c = Creators(size="wrong_type")
        c = Creators(size=(creators, attr))
        assert c.actual_creator_profiles.shape == (creators, attr)
        c = Creators(actual_creator_profiles=np.random.randint(5, size=(creators, attr)))
        assert c.actual_creator_profiles.shape == (creators, attr)
        # can't normalize a vector that isn't a matrix
        c = Creators(actual_creator_profiles=[[1, 2, 3]])

        new_items = c.generate_items()
        assert new_items.shape[0] <= 1 # there is only one creator
        assert new_items.shape[1] == 3