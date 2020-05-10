import numpy as np
import pandas as pd


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
