import numpy as np
import pytest
import scipy.sparse as sp
import test_helpers
import trecs.matrix_ops as mo


class TestMatrixOps:
    def test_normalize_matrix(self):
        # matrix that already has norm 1 columns/rows
        mat_1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        np.testing.assert_array_almost_equal(mat_1, mo.normalize_matrix(mat_1))
        np.testing.assert_array_almost_equal(mat_1, mo.normalize_matrix(mat_1, axis=0))

        # matrix with norm 2 columns/rows
        mat_2 = np.array([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]])
        np.testing.assert_array_almost_equal(mat_1, mo.normalize_matrix(mat_2))
        np.testing.assert_array_almost_equal(mat_1, mo.normalize_matrix(mat_2, axis=0))

        # check norm of all rows equals 1 after normalization
        mat_3 = np.arange(16).reshape((4, 4))
        normalized = mo.normalize_matrix(mat_3, axis=1)
        assert (np.linalg.norm(normalized, axis=1) == 1).all()

        # add additional column
        mat_4 = np.hstack([mat_1, np.array([1, 0, 0, 0]).reshape(-1, 1)])
        np.testing.assert_array_almost_equal(mat_4, mo.normalize_matrix(mat_4, axis=0))

    def test_normalize_vector(self):
        vec = np.array([3, 4])
        unit_vec = mo.normalize_matrix(vec)
        correct_unit_vec = np.array([[3 / 5, 4 / 5]])
        np.testing.assert_array_almost_equal(unit_vec, correct_unit_vec)

    def test_contains_row(self):
        mat = np.arange(16).reshape((4, 4))
        assert mo.contains_row(mat, [0, 1, 2, 3])
        assert not mo.contains_row(mat, [3, 2, 1, 0])

    def test_slerp(self):
        # rotate unit vectors 45 degrees
        mat1 = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
        mat2 = np.array([[1, 0], [0, -1], [-1, 0], [0, 1]])
        rotated = mo.slerp(mat1, mat2, perc=0.5)
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
        rotated = mo.slerp(mat1_big, mat2, perc=0.5)
        np.testing.assert_array_almost_equal(rotated, 2 * correct_rotation)

        # only rotate 5% and then verify that the angle between each row of the
        # resulting matrix and the target matrix is 0.95 * 90
        rotated = mo.slerp(mat1, mat2, perc=0.05)
        theta = np.arccos((rotated * mat2).sum(axis=1))
        test_helpers.assert_equal_arrays(theta, np.repeat([np.pi / 2 * 0.95], 4))

        # rotate a unit vector 45 degrees
        vec1 = np.array([np.sqrt(2) / 2, np.sqrt(2) / 2])
        vec2 = np.array([np.sqrt(2) / 2, -np.sqrt(2) / 2])
        correct_rotation = np.array([[1, 0]])
        rotated = mo.slerp(vec1, vec2, perc=0.5)
        test_helpers.assert_equal_arrays(rotated, correct_rotation)

    def test_sparse_and_dense(self):
        x = np.zeros((3, 4))
        y = np.ones((3, 5))
        assert mo.all_dense(x, y)
        assert mo.all_dense([x, y])
        assert mo.all_dense([x, y], x, y)

        y = sp.csr_matrix(y)

        assert not mo.all_dense(x, y)
        assert not mo.all_dense([x, y])
        assert not mo.all_dense([x, y], y)
        assert mo.any_dense(x, y)
        assert mo.any_dense([x, y])

        x = sp.csr_matrix(x)
        assert not mo.all_dense(x, y)
        assert not mo.all_dense([x, y])
        assert not mo.any_dense(x, y)
        assert not mo.any_dense([x, y])

        z = "hello"
        with pytest.raises(TypeError):
            mo.all_dense(z)

        with pytest.raises(TypeError):
            mo.all_dense(z)

        with pytest.raises(TypeError):
            mo.all_dense(z)

    def test_hstack(self):
        x = np.zeros((3, 4))
        y = np.ones((3, 5))
        z = mo.hstack([x, y])
        assert isinstance(z, np.ndarray)

        y = sp.csr_matrix(y)
        z = mo.hstack([x, y])
        # if at least one argument is dense, all other arguments should be
        # converted to dense, and the result returned should be dense
        assert isinstance(z, np.ndarray)
        # if all arguments are sparse, then the return value should be
        # sparse
        x = sp.coo_matrix(x)
        z = mo.hstack([x, y])
        assert isinstance(z, sp.spmatrix)

    def test_add_empty_cols(self):
        x = np.ones((3, 5))
        y = mo.add_empty_cols(x, 5)
        assert isinstance(y, np.ndarray)
        assert y.shape == (3, 10)
        assert (y[:, 5:] == 0).all()

        # test sparse matrix
        x = sp.csr_matrix(x)
        y = mo.add_empty_cols(x, 5)
        assert isinstance(y, sp.spmatrix)
        assert y.shape == (3, 10)
        assert (y.toarray()[:, 5:] == 0).all()

    def test_inner_product(self):
        x = np.ones((3, 5))
        y = np.eye(5) * 2
        z = mo.inner_product(x, y, normalize_users=False, normalize_items=False)
        assert isinstance(z, np.ndarray)
        correct_answer = np.ones((3, 5)) * 2
        np.testing.assert_array_almost_equal(z, correct_answer)

        # if one array is dense, the final return value should be dense
        y = sp.csr_matrix(y)
        z = mo.inner_product(x, y, normalize_users=False, normalize_items=False)
        assert isinstance(z, np.ndarray)
        np.testing.assert_array_almost_equal(z, correct_answer)

        # if both arrays are sparse, the final return value should be sparse
        x = sp.csr_matrix(x)
        z = mo.inner_product(x, y, normalize_users=False, normalize_items=False)
        assert isinstance(z, sp.spmatrix)
        correct_answer = sp.csr_matrix(correct_answer)
        np.testing.assert_array_almost_equal(z.A, correct_answer.A)

    def test_sparse_dot(self):
        x = np.ones((3, 5))
        y = sp.csr_matrix(np.eye(5) * 2)
        with pytest.raises(TypeError):
            mo.sparse_dot(x, y)

    def test_argmax(self):
        x = np.eye(10)
        y = sp.csr_matrix(x.copy())

        z1 = mo.argmax(x, axis=0)
        z2 = mo.argmax(y, axis=0)
        np.testing.assert_array_equal(z1, z2)

        z1 = mo.argmax(x, axis=1)
        z2 = mo.argmax(y, axis=1)
        np.testing.assert_array_equal(z1, z2)
