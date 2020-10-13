"""Various utility functions, mainly for matrices and input validation"""
import numpy as np
from rec.components import Users

# Common input validation functions
def validate_user_item_inputs(  # pylint: disable=too-many-arguments
    num_users,
    num_items,
    items_hat,
    users_hat,
    items,
    users,
    default_num_users,
    default_num_items,
    num_attributes=None,
):
    """ Validate that the user and item matrices passed in look
        correct
    """
    if not is_array_valid_or_none(items_hat, ndim=2):
        raise ValueError("items matrix must be a 2D matrix or None")
    if not is_array_valid_or_none(users_hat, ndim=2):
        raise ValueError("users matrix must be a 2D matrix or None")

    # check attributes matching for users_hat and items_hat
    num_attrs_vals = non_none_values(
        getattr(users_hat, "shape", [None, None])[1],
        getattr(items_hat, "shape", [None])[0],
        num_attributes,
    )
    if len(num_attrs_vals) > 1:
        raise ValueError(
            "User representation and item representation matrices are not "
            "compatible with each other"
        )

    num_items_vals = non_none_values(
        getattr(items_hat, "shape", [None, None])[1],
        getattr(items, "shape", [None, None])[1],
        num_items,
    )

    # if users is a Users object, we check to make sure it contains consistent

    num_users_vals = non_none_values(
        getattr(users, "shape", [None])[0], getattr(users_hat, "shape", [None])[0], num_users
    )

    if isinstance(users, Users):
        num_users_vals = num_users_vals.union(
            non_none_values(
                get_first_valid(
                    getattr(users.actual_user_scores, "shape", [None])[0],
                    getattr(users.actual_user_profiles, "shape", [None])[0],
                )
            )
        )

    if len(num_users_vals) == 0:  # number of users not specified anywhere
        num_users = default_num_users
    elif len(num_users_vals) == 1:
        num_users = list(num_users_vals)[0]  # should be the single number of users
    else:
        raise ValueError("Number of users is not the same across inputs")
    if len(num_items_vals) == 0:  # number of items not specified anywhere
        num_items = default_num_items
    elif len(num_items_vals) == 1:
        num_items = list(num_items_vals)[0]  # should be the single number of users
    else:
        raise ValueError("Number of items is not the same across inputs")
    return num_users, num_items


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


def non_none_values(*args):
    """ Return True if all of the (non-None) elements are equal
    """
    return set(filter(None, args))


def all_none(*args):
    """ Return True if all arguments passed in are None. """
    return all(a is None for a in args)
