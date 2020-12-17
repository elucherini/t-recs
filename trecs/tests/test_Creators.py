import numpy as np
from trecs.components import Creators
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
        with pytest.raises(ValueError):
            c = Creators(creation_probability=30)
        with pytest.raises(ValueError):
            c = Creators(creation_probability=-6)
        c = Creators(size=(creators, attr))
        assert c.actual_creator_profiles.shape == (creators, attr)
        c = Creators(actual_creator_profiles=np.random.randint(5, size=(creators, attr)))
        assert c.actual_creator_profiles.shape == (creators, attr)

    def test_item_creation(self):
        # 10 users, with 5 attributes each
        profiles = np.random.uniform(size=(10, 5))
        c = Creators(actual_creator_profiles=profiles)

        for _ in range(5):
            new_items = c.generate_items()
            assert new_items.shape[1] <= 10  # number of items created should be at most 10
            assert new_items.shape[0] == 5  # number of attributes should always be 5

    def test_invalid_item_creation(self):
        profiles = -1 * np.random.uniform(size=(10, 5))
        c = Creators(actual_creator_profiles=profiles)

        with pytest.raises(ValueError):
            c.generate_items()
