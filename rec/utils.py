import numpy as np


"""
'' Normalize matrix
'' @matrix: matrix to normalize
'' @axis: 1 -> rows, 0 -> columns
"""


def normalize_matrix(matrix, axis=1):
    """ Normalize a matrix so that each row vector has a Euclidean norm of 1.
        If a vector is passed in, we treat it as a matrix with a single row.
    """
    if len(matrix.shape) == 1:
        # turn vector into matrix with one row
        matrix = matrix[:, np.newaxis]
    divisor = np.linalg.norm(matrix, axis=1)[:, np.newaxis]
    # only normalize where divisor is not zero
    result = np.divide(matrix, divisor, out=np.zeros(matrix.shape), where=divisor != 0)
    return result


def contains_row(matrix, row):
    """ Check if a numpy matrix contains a row with the same values as the 
        variable `row`.
    """
    return (matrix == row).all(1).any()


def slerp(mat1, mat2, t=0.05):
    """ Implements `spherical linear interpolation`_. Takes each row vector in
        mat1 and rotates it in the direction of the corresponding row vector in
        mat2. The angle of rotation is `(t * the angle between the two row
        vectors)`. i.e., when `t=0.05`, each row vector
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

            t: float
                Parameter in [0,1] inclusive that specifies the percentage
                of rotation.
    """
    assert t >= 0.0 and t <= 1.0
    mat1_norm = np.linalg.norm(mat1, axis=1)[:, np.newaxis]
    mat2_norm = np.linalg.norm(mat2, axis=0)[:, np.newaxis]
    # dot every user profile with its corresponding item attributes
    omega = np.arccos((mat1 / mat1_norm) * (mat2 / mat2_norm).sum(axis=1))
    # note: bad things will happen if the vectors are in exactly opposite
    # directions! this is a pathological case; we are using this function
    # to calculate user profile drift after the user selects an item.
    # but the dot product of a user profile and an item vector in opposite
    # directions is very negative, so a user should almost never select an
    # item in the opposite direction of its own profile.
    if omega == np.pi:
        # raise error if vectors are in exactly opposite directions
        raise ValueError(
            "Cannot perform spherical interpolation between vectors in opposite direction"
        )
    so = np.sin(omega)
    unit_rot = (
        np.sin((1.0 - t) * omega) / so * mat1.T + np.sin(t * omega) / so * mat2.T
    ).T
    return unit_rot * mat1_norm


def toDataFrame(data, index=None):
    import pandas as pd

    if index is None:
        df = pd.DataFrame(data)
    else:
        df = pd.DataFrame(data).set_index(index)
    return df


"""Common input validation functions"""


def is_array_valid_or_none(array, ndim=2):
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


def array_dimensions_match(array1, array2):
    """ Assuming that both arrays are defined,
        we test whether they have matching dimensions.
    """
    array1, array2 = np.asarray(array1), np.asarray(array2)
    return array1.shape == array2.shape


def is_valid_or_none(value, type):
    """Returns true if either None or of the specified type"""
    return value is None or isinstance(value, type)


def get_first_valid(*args):
    """ Returns the first value that is not None.
        If all arguments are None, the function returns None.
        This is generally used to establish priorities for num_users, num_items, etc.
    """
    for arg in args:
        if arg is not None:
            return arg
    return None


def is_equal_dim_or_none(*args):
    for i, arg in enumerate(args):
        if arg is None:
            return True
        if i + 1 < len(args) and args[i + 1] is not None and arg != args[i + 1]:
            print(arg)
            return False
    return True


"""
def validate_input(**kwargs):
    rules = kwargs.pop('rules', None)
    args = kwargs.pop('args', None)
    if len(rules) != len(args):
        return False
    for index, rule in enumerate(rules):
        if not rule(args[i]):
            return False
    return True
"""


def all_none(*arrays):
    return all(a is None for a in arrays)


import logging
import sys
import numpy as np
from abc import ABC, abstractmethod

# Abstract class for verbose mode
class VerboseMode(ABC):
    def __init__(self, name, verbose=False):
        self._logger = DebugLogger(name, verbose)

    """
    '' Toggle verbosity
    '' @toggle: True/False
    """

    def set_verbose(self, toggle):
        try:
            self._logger.set_verbose(toggle)
        except TypeError as e:
            print("set_verbose:", e)

    """
    '' Return True if verbosity is enabled,
    '' False otherwise
    """

    def is_verbose(self):
        return self._logger.is_verbose()

    def log(self, msg):
        self._logger.log(msg)


# Class to configure debug logging module
class DebugLogger:
    """
    '' @name: name of logger
    '' @level: level of logger (see documentation of logging module)
    """

    def __init__(self, name, verbose=False):
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

    """
    '' Simple test to announce logger is enabled
    """

    def _test_configured_logger(self):
        self.logger.debug("Debugging module inizialized")

    """
    '' Log at DEBUG level
    '' @message: message to log
    """

    def log(self, message):
        self.logger.debug(message)

    """
    '' Return True if debugger is enabled
    '' That is, if debugger can log DEBUG-level messages
    """

    def is_verbose(self):
        return self.logger.isEnabledFor(logging.DEBUG)

    """
    '' Enable/disable verbose
    '' @verbose: bool
    """

    def set_verbose(self, verbose=False):
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be bool, got %s type" % type(verbose))
        if verbose:
            level = logging.DEBUG
        else:
            level = logging.INFO

        self.logger.setLevel(level)
