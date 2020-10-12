"""Various utility functions, mainly for matrices and input validation"""
import logging
from abc import ABC
import numpy as np


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
    """ Normalize a matrix so that each row vector has a Euclidean norm of 1.
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
    """ Check if a numpy matrix contains a row with the same values as the
        variable `row`.
    """
    return (matrix == row).all(axis=1).any()


def slerp(mat1, mat2, perc=0.05):
    """ Implements `spherical linear interpolation`_. Takes each row vector in
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


# Common input validation functions


def is_array_valid_or_none(array, ndim=2):
    """ Return True if no array was passed in or if the array matches the
        dimensions specified
    """
    # check if array_like
    if not is_valid_or_none(array, (np.ndarray, list)):
        return False
    # check if None: this is correct and allowed
    if array is None:
        return True

    array = np.asarray(array)
    # check ndim
    if array.ndim != ndim:
        return False
    # all good
    return True


def array_dimensions_match(array1, array2, axis=None):
    """ Assuming that both arrays are defined,
        we test whether they have matching dimensions.
    """
    array1, array2 = np.asarray(array1), np.asarray(array2)
    if axis is None:
        return array1.shape == array2.shape
    return array1.shape[axis] == array2.shape[axis]


def is_valid_or_none(value, desired_type):
    """ Returns true if either None or of the specified type"""
    return value is None or isinstance(value, desired_type)


def get_first_valid(*args):
    """ Returns the first value that is not None.
        If all arguments are None, the function returns None.
        This is generally used to establish priorities for num_users, num_items, etc.
    """
    for arg in args:
        if arg is not None:
            return arg
    return None


def all_besides_none_equal(*args):
    """ Return True if all of the (non-None) elements are equal
    """
    non_none = list(filter(None, args))
    for i, arg in enumerate(non_none):
        if i + 1 < len(non_none) and arg != args[i + 1]:
            print(arg)
            return False
    return True


def all_none(*args):
    """ Return True if all arguments passed in are None. """
    return all(a is None for a in args)


class VerboseMode(ABC):
    """ Abstract class for verbose mode
    """

    def __init__(self, name, verbose=False):
        self._logger = DebugLogger(name, verbose)

    def set_verbose(self, toggle):
        """ Toggle verbosity
        """
        try:
            self._logger.set_verbose(toggle)
        except TypeError as err:
            print("set_verbose:", err)

    def is_verbose(self):
        """ Return True if verbosity is enabled, False otherwise
        """
        return self._logger.is_verbose()

    def log(self, msg):
        """ Log given message"""
        self._logger.log(msg)


class DebugLogger:
    """ Class to configure debug logging module """

    def __init__(self, name, verbose=False):
        """ Instantiate DebugLogger object
            @name: name of logger
            @level: level of logger (see documentation of logging module)
        """
        # create logger
        self.logger = logging.getLogger(name)
        if verbose:
            level = logging.DEBUG
        else:
            level = logging.INFO
        self.logger.setLevel(level)

        # create file handler and set level to debug
        self.handler = logging.FileHandler("rec.log")
        self.handler.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter("%(name)s - %(message)s")

        # add formatter to handler
        self.handler.setFormatter(formatter)

        # add handler to logger
        self.logger.addHandler(self.handler)

        # test
        self._test_configured_logger()

    def _test_configured_logger(self):
        """ Simple test to announce logger is enabled
        """
        self.logger.debug("Debugging module inizialized")

    def log(self, message):
        """ Log at DEBUG level """
        self.logger.debug(message)

    def is_verbose(self):
        """ Return True if debugger is enabled; That is, if debugger can log
            DEBUG-level messages
        """
        return self.logger.isEnabledFor(logging.DEBUG)

    def set_verbose(self, verbose=False):
        """ Enable/disable verbose """
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be bool, got %s type" % type(verbose))
        if verbose:
            level = logging.DEBUG
        else:
            level = logging.INFO

        self.logger.setLevel(level)
