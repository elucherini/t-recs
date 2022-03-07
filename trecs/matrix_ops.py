""" Common matrix operations
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import norm
from trecs.base import Component


def to_dense(arr):
    """
    Convert a sparse array to a dense numpy array. If the
    array is already a numpy array, just return it. If the
    array passed in is a list, then we recursively apply this
    method to its elements.

    Parameters
    -----------
        arr : :obj:`numpy.ndarray`, :obj:`scipy.sparse.spmatrix`, or list
            Any matrix (or list of matrices) that must be converted
            to a dense numpy array.

    Raises
    --------
        TypeError
            If the array provided is not a list, `numpy` array,
            or `scipy.sparse` matrix.

    Returns
    --------
        dense_args: tuple
    """
    if isinstance(arr, np.ndarray):
        return arr
    if isinstance(arr, list):
        return [to_dense(el) for el in arr]
    # assume it must be a `scipy.sparse.spmatrix`
    if isinstance(arr, sp.spmatrix):
        return arr.toarray()
    error_msg = (
        "Can only convert numpy matrices, scipy matrices, or "
        "lists of those elements to dense arrays"
    )
    raise TypeError(error_msg)


def all_to_dense(*args):
    """
    Returns the arguments converted to their `numpy` array
    equivalents

    Parameters
    -----------
        *args
            Arbitrary arguments (`numpy` matrices, `scipy.sparse`
            matrices, or lists containing elements of those tyeps)

    Returns
    --------
        dense_args: tuple
    """
    return (to_dense(arg) for arg in args)


def all_dense(*args):
    """
    Returns `True` if all of the arguments in the provided
    arguments (and all nested arguments) are numpy matrices.

    Parameters
    -----------
        *args
            Arbitrary arguments (`numpy` matrices, `scipy.sparse`
            matrices, or lists containing elements of those types)

    Returns
    --------
        all_dense: bool
    """
    for arg in args:
        if isinstance(arg, list):
            if not all_dense(*arg):
                return False
        elif not isinstance(arg, np.ndarray):
            if not isinstance(arg, sp.spmatrix):
                error_msg = "Argument should be only a list, numpy array, or scipy sparse matrix"
                raise TypeError(error_msg)
            return False
    return True


def all_sparse(*args):
    """
    Returns `True` if all of the arguments in the provided
    arguments (and all nested arguments) are scipy sparse matrices.

    Parameters
    -----------
        *args
            Arbitrary arguments (`numpy` matrices, `scipy.sparse`
            matrices, or lists containing elements of those types)

    Returns
    --------
        all_sparse: bool
    """
    for arg in args:
        if isinstance(arg, list):
            if not all_sparse(*arg):
                return False
        elif not isinstance(arg, sp.spmatrix):
            if not isinstance(arg, np.ndarray):
                error_msg = "Argument should be only a list, numpy array, or scipy sparse matrix"
                raise TypeError(error_msg)
            return False
    return True


def any_dense(*args):
    """
    Returns `True` if any of the arguments in the provided
    arguments (or any nested arguments) are numpy matrices.

    Parameters
    -----------
        *args
            Arbitrary arguments

    Returns
    --------
        all_dense: bool
    """
    for arg in args:
        if isinstance(arg, list):
            if any_dense(*arg):
                return True
        elif isinstance(arg, np.ndarray):
            return True
        elif not isinstance(arg, sp.spmatrix):
            error_msg = "Argument should be only a list, numpy array, or scipy sparse matrix"
            raise TypeError(error_msg)
    return False


def extract_value(arg):
    """
    Extracts the value from an argument if it is a
    :class:`~base.base_components.Component`; otherwise, returns
    the argument itself.

    Parameters
    -----------
        arg: any
            Arbitrary variable. If `arg` is a
            :class:`~base.base_components.Component`, then its
            `current_state` will be returned. Otherwise, we return
            arg.

    Parameters
    -----------
        *args
            Arbitrary arguments

    Returns
    --------
        values: single value or list
    """
    if isinstance(arg, Component):
        return arg.value
    return arg


def extract_values(*args):
    """
    Wrapper around `extract_value`; iteratively applies that method to all items
    in a list. If only one item was passed in, then we return that one item's
    value; if multiple items were passed in, we return a list of the corresponding
    item values.

    """
    processed = [extract_value(arg) for arg in args]
    if len(processed) == 1:
        return processed[0]
    return processed


def transpose(mat):
    """
    Performs a simple matrix transpose.

    Parameters
    -----------
        mat: :obj:`numpy.ndarray`, :obj:`scipy.sparse.spmatrix`, or
        `~base.base_components.Component`
            Matrix to transpose.

    Returns
    --------
        transpose: :obj:`numpy.ndarray` or :obj:`scipy.sparse.spmatrix`
    """
    mat = extract_value(mat)
    return mat.T


def generic_matrix_op(dense_fn, sparse_fn, *matrix_args, **kwargs):
    """
    Applies the supplied dense function if the arguments provided are
    numpy arrays, or applies the supplied sparse function if the
    arguments provided are sparse matrices. Accepts optional keyword
    arguments which are passed to the sparse/dense functions. If at least
    one of the arguments in `matrix_args` is a numpy array, then all of
    the `matrix_args` will be converted to numpy arrays and
    `dense_fn` will be called; otherwise, if all of the arguments in
    `matrix_args` are `scipy` sparse arrays, then `sparse_fn` will
    be called.

    Parameters
    -----------
        dense_fn: callable
            A function that is intended to be called on `numpy`
            arrays.

        sparse_fn: callable
            A function that is intended to be called on `scipy`
            sparse matrices.

        *args
            Arguments for `dense_fn` or `sparse_fn`.

        **kwargs
            Optional keyword arguments that will be passed
            to `dense_fn` or `sparse_fn`.

    Returns
    --------
        matrix: :obj:`numpy.ndarray` or :obj:`scipy.sparse.spmatrix`
    """
    # if arguments are Components, we extract their matrix values
    matrix_args = extract_values(matrix_args)
    if all_dense(*matrix_args):
        return dense_fn(*matrix_args, **kwargs)
    if any_dense(*matrix_args):
        return dense_fn(*all_to_dense(*matrix_args), **kwargs)
    return sparse_fn(*matrix_args, **kwargs)


def hstack(matrix_list):
    """
    Wrapper for `numpy.hstack` and `scipy.sparse.hstack`. Horizontally
    concatenates matrices and returns a `numpy` array or `scipy` sparse
    matrix, depending on the arguments provided.

    Parameters
    -----------
        matrix_list: list
            List of matrices to be concatenated

    Returns
    --------
        matrix: :obj:`numpy.ndarray` or :obj:`scipy.sparse.spmatrix`
            Resulting sparse matrix or dense matrix.
    """
    return generic_matrix_op(np.hstack, sp.hstack, matrix_list)


def vstack(matrix_list):
    """
    Wrapper for `numpy.vstack` and `scipy.sparse.vstack`. Vertically
    concatenates matrices and returns a `numpy` array or `scipy` sparse
    matrix, depending on the arguments provided.

    Parameters
    -----------
        matrix_list: list
            List of matrices to be concatenated

    Returns
    --------
        matrix: :obj:`numpy.ndarray` or :obj:`scipy.sparse.spmatrix`
            Resulting sparse matrix or dense matrix.
    """
    return generic_matrix_op(np.vstack, sp.vstack, matrix_list)


def sparse_argmax(matrix, axis=None):
    """
    By default, the implementation of `argmax` in `scipy` returns
    a `numpy.matrix` - no good! Here, we force the output of the
    argmax function to be a 1D array.

    Parameters
    -----------
        matrix: `scipy.sparse.spmatrix`
            Matrix from which we want the argmax.

        axis: int
            Axis along which to take the argmax

    Returns
    --------
        array: :obj:`numpy.ndarray`
            Flattened 1D array of the row/col arg max.
    """
    mat = matrix.argmax(axis=axis)
    return np.squeeze(np.asarray(mat))


def argmax(matrix, axis=None):
    """
    Method that "standardizes" the output of
    `numpy.argmax` and `scipy.sparse.argmax`.

    Parameters
    -----------
        matrix: `scipy.sparse.spmatrix`
            Matrix from which we want the argmax.

        axis: int
            Axis along which to take the argmax

    Returns
    --------
        array: :obj:`numpy.ndarray`
            Flattened 1D array of the row/col arg max.
    """
    return generic_matrix_op(np.argmax, sparse_argmax, matrix, axis=axis)


def top_k_indices(matrix, k, random_state):
    """
    Given a matrix of values, we return the indices of the greatest values,
    per-row, where the indices will appear in descending order of associated
    value. This method is efficient in that we use argpartition to take
    the top k indices first before sorting within the top k indices. We also
    tie break randomly.

    Parameters
    -----------
        matrix: :obj:`numpy.ndarray`
            Matrix of input values.

        k: int
            Number of top indices to return per row.

        random_state: :obj:`numpy.random.RandomState`
            Random state generator used for random tiebreaking

    Returns
    --------
        :obj:`numpy.ndarray`:
            Matrix with ``k`` columns and the same number of rows as the
            original matrix, containing the top-k indices per row, sorted
            in descending order of value
    """
    # scores are U x I; we can use argpartition to take the top k scores
    negated = -1 * matrix  # negate scores so indices go from highest to lowest
    # break ties using a random score component
    vals_tiebreak = np.zeros(negated.shape, dtype=[("score", "f8"), ("random", "f8")])
    vals_tiebreak["score"] = negated
    vals_tiebreak["random"] = random_state.random(negated.shape)
    top_k = vals_tiebreak.argpartition(k - 1, order=["score", "random"])[:, :k]
    # now we sort within the top k
    num_rows = matrix.shape[0]
    row_vector = np.repeat(np.arange(num_rows, dtype=int), k).reshape((num_rows, -1))
    # again, indices should go from highest to lowest, so we sort within the top_k
    sort_top_k = vals_tiebreak[row_vector, top_k].argsort(order=["score", "random"])
    return top_k[row_vector, sort_top_k]


def add_empty_cols(matrix, num_cols):
    """
    Adds empty columns to a matrix, which can be either a sparse
    matrix or a dense `numpy` array. These columns will be filled
    wiht zeroes.

    Parameters
    -----------
        matrix: `numpy.ndarray` or `scipy.sparse.spmatrix`
            Matrix which we want to append empty columns to.

    Raises
    --------
        TypeError
            If the matrix provided is not a `numpy` array
            or a `scipy.sparse` matrix.

    Returns
    --------
        matrix: :obj:`numpy.ndarray` or :obj:`scipy.sparse.spmatrix`
            Resulting sparse matrix or dense matrix.
    """
    if not isinstance(matrix, (np.ndarray, sp.spmatrix)):
        raise TypeError("Matrix must be a numpy array or scipy sparse matrix")
    if any_dense(matrix):
        return np.hstack([matrix, np.zeros((matrix.shape[0], num_cols))])
    else:
        return sp.hstack([matrix, sp.csr_matrix((matrix.shape[0], num_cols))])


def sparse_dot(mat1, mat2):
    """
    Returns the dot product of two sparse matrices.

    Parameters
    -----------

        mat1: :obj:`scipy.sparse.spmatrix`
            First factor of the dot product.

        mat1: :obj:`scipy.sparse.spmatrix`
            Second factor of the dot product.

    Returns
    --------
        dot_product: :obj:`scipy.sparse.spmatrix`
    """
    if not all_sparse(mat1, mat2):
        raise TypeError("sparse_inner_product can only operate on sparse matrices")
    return mat1.dot(mat2)


def inner_product(user_profiles, item_attributes, normalize_users=True, normalize_items=False):
    """
    Performs a dot product multiplication between user profiles and
    item attributes to return the scores (utility) each item possesses
    for each user. We call these matrices `user_profiles` and
    `item_attributes` but note that you could perform an arbitrary matrix
    dot product with this method.

    Parameters
    -----------

        user_profiles: :obj:`numpy.ndarray`, :obj:`scipy.sparse.spmatrix`
            First factor of the dot product, which should provide a
            representation of users.

        item_attributes: :obj:`numpy.ndarray`, :obj:`scipy.sparse.spmatrix`
            Second factor of the dot product, which should provide a
            representation of items.

    Returns
    --------
        scores: :obj:`numpy.ndarray`
    """
    if normalize_users:
        # this is purely an optimization that prevents numpy from having
        # to multiply huge numbers
        user_profiles = normalize_matrix(user_profiles, axis=1)
    if normalize_items:
        item_attributes = normalize_matrix(item_attributes.T, axis=1).T
    if user_profiles.shape[1] != item_attributes.shape[0]:
        error_message = (
            "Number of attributes in user profile matrix must match number "
            "of attributes in item profile matrix"
        )
        raise ValueError(error_message)
    scores = generic_matrix_op(np.dot, sparse_dot, user_profiles, item_attributes)
    return scores


def scale_rows_dense_fn(vector):
    """
    Returns a function that scales the rows of an
    arbitrary dense matrix by the elements of the vector.

    Parameters
    -----------

        vector: :obj:`numpy.ndarray`
            Should have length equal to the number of rows
            of the matrix passed in to the return function.
    """
    return lambda m: np.multiply(m, vector[:, np.newaxis])


def scale_rows_sparse_fn(vector):
    """
    Returns a function that scales the rows of an
    arbitrary sparse matrix by the elements of the vector.

    Parameters
    -----------

        vector: :obj:`numpy.ndarray`
            Should have length equal to the number of rows
            of the matrix passed in to the return function.
    """
    return lambda m: sparse_dot(sp.diags(vector), m)


def normalize_matrix(matrix, axis=1):
    """
    Normalize a matrix so that each row vector has a Euclidean norm of 1.
    If a vector is passed in, we treat it as a matrix with a single row.
    """
    if len(matrix.shape) == 1:
        # turn vector into matrix with one row
        matrix = matrix[np.newaxis, :]
    if axis == 0:
        return normalize_matrix(matrix.T, axis=1).T
    divisor = generic_matrix_op(np.linalg.norm, norm, matrix, axis=axis)
    divisor[divisor == 0] = -1  # sentinel value to avoid division by zero in next step
    # row scale by diagonal matrix
    divisor = 1 / divisor
    divisor[divisor == -1] = 0

    dense_fn, sparse_fn = scale_rows_dense_fn(divisor), scale_rows_sparse_fn(divisor)
    return generic_matrix_op(dense_fn, sparse_fn, matrix)


def contains_row(matrix, row):
    """Check if a numpy matrix contains a row with the same values as the
    variable `row`.
    """
    return (matrix == row).all(axis=1).any()


def slerp(mat1, mat2, perc=0.05):
    """Implements `spherical linear interpolation`_. Takes each row vector in
    mat1 and rotates it in the direction of the corresponding row vector in
    mat2. The angle of rotation is `(perc * the angle between the two row
    vectors)`. i.e., when `perc=0.05`, each row vector
    in mat1 is rotated with an angle equal to 5% of the total angle
    between it and the corresponding row vector in mat2. The matrix returned
    will have row vectors that each have the same norm as mat1, but pointing
    in different directions.

    .. _`spherical linear interpolation`: https://en.wikipedia.org/wiki/Slerp

    Parameters
    -----------

        mat1: :obj:`numpy.ndarray` or list
            Matrix whose row vectors will be rotated in the direction of
            mat2's row vectors.

        mat2: :obj:`numpy.ndarray` or list
            Matrix that should have the same dimensions at mat1.

        perc: float
            Parameter in [0,1] inclusive that specifies the percentage
            of rotation.
    """
    # in case Components were passed
    mat1, mat2 = extract_values(mat1, mat2)
    if perc < 0 or perc > 1.0:
        raise ValueError("Percentage rotation must be between 0 and 1.")
    if not mat1.shape == mat2.shape:
        # arrays should have same dimension
        raise ValueError("Matrices must have the same shape for rows to be rotated")
    if len(mat1.shape) == 1:
        # turn vector into matrix with one row
        mat1 = mat1[np.newaxis, :]
        mat2 = mat2[np.newaxis, :]
    mat1_length = np.linalg.norm(mat1, axis=1)[:, np.newaxis]
    mat2_length = np.linalg.norm(mat2, axis=1)[:, np.newaxis]
    mat1_norm, mat2_norm = mat1 / mat1_length, mat2 / mat2_length
    row_dot_product = (mat1_norm * mat2_norm).sum(axis=1)
    # dot every user profile with its corresponding item attributes
    omega = np.arccos(row_dot_product)
    # note: bad things will happen if the vectors are in exactly opposite
    # directions! this is a pathological case; we are using this function
    # to calculate user profile drift after the user selects an item.
    # but the dot product of a user profile and an item vector in opposite
    # directions is very negative, so a user should almost never select an
    # item in the opposite direction of its own profile.
    if (omega == np.pi).any():
        # raise error if vectors are in exactly opposite directions
        raise ValueError(
            "Cannot perform spherical interpolation between vectors in opposite direction"
        )
    sin_o = np.sin(omega)
    unit_rot = (
        np.sin((1.0 - perc) * omega) / sin_o * mat1_norm.T
        + np.sin(perc * omega) / sin_o * mat2_norm.T
    ).T
    return unit_rot * mat1_length
