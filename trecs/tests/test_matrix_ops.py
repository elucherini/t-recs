import numpy as np
import test_helpers
from trecs.matrix_ops import normalize_matrix, contains_row, slerp


class TestMatrixOps:
    def test_normalize_matrix(self):
        # matrix that already has norm 1 columns/rows
        mat_1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        test_helpers.assert_equal_arrays(mat_1, normalize_matrix(mat_1))
        test_helpers.assert_equal_arrays(mat_1, normalize_matrix(mat_1, axis=0))

        # matrix with norm 2 columns/rows
        mat_2 = np.array([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]])
        test_helpers.assert_equal_arrays(mat_1, normalize_matrix(mat_2))
        test_helpers.assert_equal_arrays(mat_1, normalize_matrix(mat_2, axis=0))

        # check norm of all rows equals 1 after normalization
        mat_3 = np.arange(16).reshape((4, 4))
        normalized = normalize_matrix(mat_3, axis=1)
        assert (np.linalg.norm(normalized, axis=1) == 1).all()

    def test_normalize_vector(self):
        vec = np.array([3, 4])
        unit_vec = normalize_matrix(vec)
        correct_unit_vec = np.array([[3 / 5, 4 / 5]])
        test_helpers.assert_equal_arrays(unit_vec, correct_unit_vec)

    def test_contains_row(self):
        mat = np.arange(16).reshape((4, 4))
        assert contains_row(mat, [0, 1, 2, 3])
        assert not contains_row(mat, [3, 2, 1, 0])

    def test_slerp(self):
        # rotate unit vectors 45 degrees
        mat1 = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
        mat2 = np.array([[1, 0], [0, -1], [-1, 0], [0, 1]])
        rotated = slerp(mat1, mat2, perc=0.5)
        correct_rotation = np.array(
            [
                [np.sqrt(2) / 2, np.sqrt(2) / 2],
                [np.sqrt(2) / 2, -np.sqrt(2) / 2],
                [-np.sqrt(2) / 2, -np.sqrt(2) / 2],
                [-np.sqrt(2) / 2, np.sqrt(2) / 2],
            ]
        )
        # there may be imprecision due to floating point errors
        np.testing.assert_array_almost_equal(rotated, correct_rotation)

        # increase norm of vectors and check that norm of rotated vectors
        # does not change
        mat1_big = np.array([[0, 2], [2, 0], [0, -2], [-2, 0]])
        rotated = slerp(mat1_big, mat2, perc=0.5)
        np.testing.assert_array_almost_equal(rotated, 2 * correct_rotation)

        # only rotate 5% and then verify that the angle between each row of the
        # resulting matrix and the target matrix is 0.95 * 90
        rotated = slerp(mat1, mat2, perc=0.05)
        theta = np.arccos((rotated * mat2).sum(axis=1))
        test_helpers.assert_equal_arrays(theta, np.repeat([np.pi / 2 * 0.95], 4))

        # rotate a unit vector 45 degrees
        vec1 = np.array([np.sqrt(2) / 2, np.sqrt(2) / 2])
        vec2 = np.array([np.sqrt(2) / 2, -np.sqrt(2) / 2])
        correct_rotation = np.array([[1, 0]])
        rotated = slerp(vec1, vec2, perc=0.5)
        test_helpers.assert_equal_arrays(rotated, correct_rotation)
