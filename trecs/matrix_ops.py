""" Common matrix operations
"""
import numpy as np
from scipy.stats import norm


def calc_choice_probabilities(image_matrices, values):
    """
    Scores items according to divisive normalization.

    Parameters
    -----------
        image_matrices: :obj:`numpy.ndarray`
            Image matrices that are used for transforming covariance matrix
            of errors into covariance matrix of error differences. Dimension:
            :math:`|U|\times|I|-1\times|I|`, where :math:`|I|` is the total number
            of items and :math:`|U|` is the total number of users. Note:
            these image matrices should correspond to a particular candidate
            choice for each user.

        values: :obj:`numpy.ndarray`
            Normalized values (see divisive_normalization) of each item
            for each user. Should be dimension :math: `|I|\times|U|`.

    """
    imat_T = np.transpose(image_matrices, axes=(0, 2, 1))
    [x, w] = np.polynomial.hermite.hermgauss(100)

    # TODO: verify correctness / understand mathematical operations
    c = np.tensordot(imat_T, values, axes=([1, 0]))
    c_T = np.transpose(c, axes=(0, 2, 1))
    vi = c_T.diagonal()

    z1 = np.multiply(-(2 ** 0.5), vi)
    z2 = np.multiply(-(2 ** 0.5), x)
    zz = [z1 - e for e in z2]

    aa = np.prod(norm.cdf(zz), axis=1)
    # Pi have been validated
    p = np.divide(np.sum(np.multiply(w.reshape(100, 1), aa), axis=0), np.pi ** 0.5)
    return p


def divisive_normalization(user_item_scores, sigma=0.0, omega=0.2376, beta=0.9739):
    """
    Scores items according to divisive normalization.

    Parameters
    -----------

        user_item_scores: :obj:`array_like`
            The element at index :math:`i,j` should represent user :math:`i`'s
            context-independent value for item :math:`j`.
            Dimension: :math:`|U|\times|I|`

    Returns
    --------
        probs: :obj:`numpy.ndarray`
            Probabilities of a user making a particular choice. All probabilities
            for a given row will sum up to 1.
    """
    denom = sigma + np.multiply(omega, np.linalg.norm(user_item_scores, ord=beta, axis=1))
    normed_values = np.divide(user_item_scores.T, denom)  # now |I| x |U|

    # precalculate image matrices for choices
    num_choices = normed_values.shape[0]
    eye_mats = np.identity(num_choices - 1)
    image_mats = np.empty((num_choices, num_choices - 1, num_choices))
    negative_one_vec = -1 * np.ones((num_choices - 1, 1))

    for i in range(num_choices):
        mat_parts = (eye_mats[:, :i], negative_one_vec, eye_mats[:, i:])
        image_mats[i] = np.concatenate(mat_parts, axis=1)

    # naive approach: iterate over all possible choice indices
    probs = np.zeros(normed_values.shape)
    for i in range(normed_values.shape[0]):
        choices = (np.ones(normed_values.shape[1]) * i).astype(int)
        probs[i, :] = calc_choice_probabilities(image_mats[choices], normed_values)
    return probs.T


def sample_from_items(user_item_probabilities):
    """
    Given a matrix of user-item selection probabilities, where each
    row is a vector of item probabilities for a particular user, returns the
    indices of each user's chosen item.

    Parameters
    -----------
        user_item_probabilities: :obj:`array_like`
            The element at index :math:`i,j` should represent user :math:`i`'s
            probability of choosing item :math:`j`. (Note that item `j` in one
            row may not refer to the same item as item `j` in another row, since
            each user may be shown different items in their recommendation
            set.)
            Dimension: :math:`|U|\times|I|`
    """
    num_users = user_item_probabilities.shape[0]
    num_items = user_item_probabilities.shape[1]
    chosen_indices = np.zeros(num_users)
    for u in range(num_users):
        chosen_indices[u] = np.random.sample(num_items, p=user_item_probabilities[u])
    return chosen_indices


def inner_product(user_profiles, item_attributes, normalize=True):
    """
    Performs a dot product multiplication between user profiles and
    item attributes to return the scores (utility) each item possesses
    for each user. We call these matrices `user_profiles` and
    `item_attributes` but note that you could perform an arbitrary matrix
    dot product with this method.

    Parameters
    -----------

        user_profiles: :obj:`array_like`
            First factor of the dot product, which should provide a
            representation of users.

        item_attributes: :obj:`array_like`
            Second factor of the dot product, which should provide a
            representation of items.

    Returns
    --------
        scores: :obj:`numpy.ndarray`
    """
    if normalize:
        # this is purely an optimization that prevents numpy from having
        # to multiply huge numbers
        user_profiles = normalize_matrix(user_profiles, axis=1)
    assert user_profiles.shape[1] == item_attributes.shape[0]
    scores = np.dot(user_profiles, item_attributes)
    return scores


def normalize_matrix(matrix, axis=1):
    """Normalize a matrix so that each row vector has a Euclidean norm of 1.
    If a vector is passed in, we treat it as a matrix with a single row.
    """
    if len(matrix.shape) == 1:
        # turn vector into matrix with one row
        matrix = matrix[np.newaxis, :]
    divisor = np.linalg.norm(matrix, axis=axis)[:, np.newaxis]
    # only normalize where divisor is not zero
    result = np.divide(matrix, divisor, out=np.zeros(matrix.shape), where=divisor != 0)
    return result


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

        mat1: numpy.ndarray or list
            Matrix whose row vectors will be rotated in the direction of
            mat2's row vectors.

        mat2: numpy.ndarray or list
            Matrix that should have the same dimensions at mat1.

        perc: float
            Parameter in [0,1] inclusive that specifies the percentage
            of rotation.
    """
    assert 0 <= perc <= 1.0
    assert mat1.shape == mat2.shape  # arrays should have same dimension
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
