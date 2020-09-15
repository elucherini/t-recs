import numpy as np
import test_utils
from rec.utils import (
    normalize_matrix,
    contains_row
)

class TestUtils:
    def test_normalize_matrix(self):
        # matrix that already has norm 1 columns/rows
        mat_1 = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        test_utils.assert_equal_arrays(mat_1, normalize_matrix(mat_1))
        test_utils.assert_equal_arrays(mat_1, normalize_matrix(mat_1, axis=0))
        
        # matrix with norm 2 columns/rows
        mat_2 = np.array([
            [2, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 2]
        ])
        test_utils.assert_equal_arrays(mat_1, normalize_matrix(mat_2))
        test_utils.assert_equal_arrays(mat_1, normalize_matrix(mat_2, axis=0))

        # check norm of all rows equals 1 after normalization
        mat_3 = np.arange(16).reshape((4,4))
        normalized = normalize_matrix(mat_3, axis=1)
        assert (np.linalg.norm(normalized, axis=1) == 1).all()

    def test_normalize_vector(self):
        vec = np.array([3, 4])
        unit_vec = normalize_matrix(vec)
        correct_unit_vec = np.array(
            [[3/5, 4/5]
        ])
        test_utils.assert_equal_arrays(unit_vec, correct_unit_vec)

    def test_contains_row(self):
        mat = np.arange(16).reshape((4, 4))
        assert contains_row(mat, [0, 1, 2, 3])
        assert not contains_row(mat, [3, 2, 1, 0])
