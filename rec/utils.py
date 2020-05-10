import numpy as np


'''
'' Normalize matrix
'' @matrix: matrix to normalize
'' @axis: 1 -> rows, 0 -> columns
'''
def normalize_matrix(matrix, axis=1):
    divisor = matrix.sum(axis=axis)[:, None]
    result = np.divide(matrix, divisor,
                       out=np.zeros(matrix.shape, dtype=float), where=divisor!=0)
    return result

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
        if i + 1 < len(args) and args[i+1] is not None and arg != args[i+1]:
            print(arg)
            return False
    return True
'''
def validate_input(**kwargs):
    rules = kwargs.pop('rules', None)
    args = kwargs.pop('args', None)
    if len(rules) != len(args):
        return False
    for index, rule in enumerate(rules):
        if not rule(args[i]):
            return False
    return True
'''

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

    '''
    '' Toggle verbosity
    '' @toggle: True/False
    '''
    def set_verbose(self, toggle):
        try:
            self._logger.set_verbose(toggle)
        except TypeError as e:
            print("set_verbose:", e)

    '''
    '' Return True if verbosity is enabled,
    '' False otherwise
    '''
    def is_verbose(self):
        return self._logger.is_verbose()

    def log(self, msg):
        self._logger.log(msg)


# Class to configure debug logging module
class DebugLogger():
    '''
    '' @name: name of logger
    '' @level: level of logger (see documentation of logging module)
    '''
    def __init__(self, name, verbose=False):
        # create logger
        self.logger = logging.getLogger(name)
        if verbose:
            level = logging.DEBUG
        else:
            level = logging.INFO
        self.logger.setLevel(level)

        # create file handler and set level to debug
        self.handler = logging.FileHandler('rec.log')
        self.handler.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter('%(name)s - %(message)s')

        # add formatter to handler
        self.handler.setFormatter(formatter)

        # add handler to logger
        self.logger.addHandler(self.handler)

        # test
        self._test_configured_logger()

    '''
    '' Simple test to announce logger is enabled
    '''
    def _test_configured_logger(self):
        self.logger.debug("Debugging module inizialized")

    '''
    '' Log at DEBUG level
    '' @message: message to log
    '''
    def log(self, message):
        self.logger.debug(message)

    '''
    '' Return True if debugger is enabled
    '' That is, if debugger can log DEBUG-level messages
    '''
    def is_verbose(self):
        return self.logger.isEnabledFor(logging.DEBUG)

    '''
    '' Enable/disable verbose
    '' @verbose: bool
    '''
    def set_verbose(self, verbose=False):
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be bool, got %s type" % type(verbose))
        if verbose:
            level = logging.DEBUG
        else:
            level = logging.INFO

        self.logger.setLevel(level)


# Test
if __name__ == '__main__':
    class A(VerboseMode):
        def __init__(self):
            self.a = 2
            super().__init__(__name__, True)
            self.log("hi")

    a = A()
    a.log("verbose")
    try:
        a.set_verbose('a')
    except TypeError as e:
        print(e)

    a.log("verbose")
