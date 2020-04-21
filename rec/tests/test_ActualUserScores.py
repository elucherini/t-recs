import numpy as np
from rec import ActualUserScores
import test_utils

class TestActualUserScores:
    def test_ActualUserScores(self, items=10, attr=5, users=6, expand_items_by=2):
        # random binary item_representation
        item_repr = np.random.randint(2, size=(attr, items))
        s = ActualUserScores(users, item_repr)
        test_utils.assert_equal_arrays(s.actual_scores,
                                   s.get_actual_user_scores())

        # expand items
        items += expand_items_by
        item_repr = np.concatenate((item_repr,
                        np.random.randint(2, size=(attr, expand_items_by))), axis=1)
        assert(item_repr.shape[0] == attr)
        assert(item_repr.shape[1] == items)
        s.expand_items(item_repr)
        test_utils.assert_equal_arrays(s.get_actual_user_scores(),
                                    s.actual_scores)
