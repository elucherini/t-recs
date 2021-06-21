"""Various utility functions, mainly used for input validation"""
import numpy as np
from scipy.sparse import csr_matrix

# Common input validation functions
def check_consistency(  # pylint: disable=too-many-arguments
    num_users=None,
    num_items=None,
    users_hat=None,
    items_hat=None,
    users=None,
    items=None,
    user_item_scores=None,
    default_num_users=None,
    default_num_items=None,
    default_num_attributes=None,
    num_attributes=None,
    attributes_must_match=True,
):
    """Validate that the inputs to the recommender system are consistent
    based on their dimensions. Furthermore, if all of the inputs
    are consistent, we return the number of users and items that are inferred
    from the inputs, or fall back to a provided default number.

    Parameters
    -----------

    num_users: int, optional
        An integer representing the number of users in the system

    num_items: int, optional
        An integer representing the number of items in the system

    users_hat: :obj:`numpy.ndarray`, optional
        A 2D matrix whose first dimension should be equal to the number of
        users in the system. Typically this matrix refers to the system's
        internal representation of user profiles, not the "true" underlying
        user profiles, which are unknown to the system.

    items_hat: :obj:`numpy.ndarray`, optional
        A 2D matrix whose second dimension should be equal to the number of
        items in the system. Typically this matrix refers to the system's
        internal representation of item attributes, not the "true" underlying
        item attributes, which are unknown to the system.

    users: :obj:`numpy.ndarray`, optional
        A 2D matrix whose first dimension should be equal to the number of
        users in the system. This is the "true" underlying user profile
        matrix.

    items: :obj:`numpy.ndarray`, optional
        A 2D matrix whose second dimension should be equal to the number of
        items in the system. This is the "true" underlying item attribute
        matrix.

    user_item_scores: :obj:`numpy.ndarray`, optional
        A 2D matrix whose first dimension is the number of users in the system
        and whose second dimension is the number of items in the system.

    default_num_users: int, optional
        If the number of users is not specified anywhere in the inputs, we return
        this value as the number of users to be returned.

    default_num_items: int, optional
        If the number of items is not specified anywhere in the inputs, we return
        this value as the number of items to be returned.'

    default_num_attributes: int, optional
        If the number of attributes in the item/user representations is not
        specified or cannot be inferred, this is the default number
        of attributes that should be used. (This applies only to users_hat
        and items_hat.)

    num_attributes: int, optional
        Check that the number of attributes per user & per item are equal to
        this specified number. (This applies only to users_hat and items_hat.)

    attributes_must_match: bool (optional, default: True)
        Check that the user and item matrices match up on the attribute dimension.
        If False, the number of columns in the user matrix and the number of
        rows in the item matrix are allowed to be different.

    Returns
    --------
        num_users: int
            Number of users, inferred from the inputs (or provided default).

        num_items: int
            Number of items, inferred from the inputs (or provided default).

        num_attributes: int (optional)
            Number of attributes per item/user profile, inferred from inputs
            (or provided default).
    """
    if not is_array_valid_or_none(items_hat, ndim=2):
        raise ValueError("items matrix must be a 2D matrix or None")
    if not is_array_valid_or_none(users_hat, ndim=2):
        raise ValueError("users matrix must be a 2D matrix or None")
    if not is_valid_or_none(num_attributes, int):
        raise TypeError("num_attributes must be an int")

    num_items_vals = non_none_values(
        getattr(items_hat, "shape", [None, None])[1],
        getattr(items, "shape", [None, None])[1],
        getattr(user_item_scores, "shape", [None, None])[1],
        num_items,
    )

    num_users_vals = non_none_values(
        getattr(users, "shape", [None])[0],
        getattr(users_hat, "shape", [None])[0],
        getattr(user_item_scores, "shape", [None])[0],
        num_users,
    )

    num_users = resolve_set_to_value(
        num_users_vals, default_num_users, "Number of users is not the same across inputs"
    )

    num_items = resolve_set_to_value(
        num_items_vals, default_num_items, "Number of items is not the same across inputs"
    )

    if attributes_must_match:
        # check attributes matching for users_hat and items_hat
        num_attrs_vals = non_none_values(
            getattr(users_hat, "shape", [None, None])[1],
            getattr(items_hat, "shape", [None])[0],
            num_attributes,
        )
        num_attrs = resolve_set_to_value(
            num_attrs_vals,
            default_num_attributes,
            "User representation and item representation matrices are not "
            "compatible with each other",
        )
        return num_users, num_items, num_attrs
    else:
        return num_users, num_items


def resolve_set_to_value(value_set, default_value, error_message):
    """Resolve a set of values to a single value, falling back to
    a default value if needed. If it is unresolvable, produce
    an error message.
    """
    if len(value_set) == 0:
        return default_value
    elif len(value_set) == 1:
        return list(value_set)[0]  # should be single value
    raise ValueError(error_message)


def is_array_valid_or_none(array, ndim=2):
    """Return True if no array was passed in or if the array matches the
    dimensions specified
    """
    # check if array_like
    if not is_valid_or_none(array, (np.ndarray, list, csr_matrix)):
        return False
    # check if None: this is correct and allowed
    if array is None:
        return True

    if not hasattr(array, "ndim"):
        array = np.asarray(array)

    # check ndim
    if array.ndim != ndim:
        return False

    # all good
    return True


def array_dimensions_match(array1, array2, axis=None):
    """Assuming that both arrays are defined,
    we test whether they have matching dimensions.
    """
    array1, array2 = np.asarray(array1), np.asarray(array2)
    if axis is None:
        return array1.shape == array2.shape
    return array1.shape[axis] == array2.shape[axis]


def is_valid_or_none(value, desired_type):
    """Returns true if either None or of the specified type"""
    return value is None or isinstance(value, desired_type)


def get_first_valid(*args):
    """Returns the first value that is not None.
    If all arguments are None, the function returns None.
    This is generally used to establish priorities for num_users, num_items, etc.
    """
    for arg in args:
        if arg is not None:
            return arg
    return None


def all_besides_none_equal(*args):
    """Return True if all of the (non-None) elements are equal"""
    non_none = list(filter(lambda x: x is not None, args))
    for i, arg in enumerate(non_none):
        if i + 1 < len(non_none) and arg != args[i + 1]:
            print(arg)
            return False
    return True


def non_none_values(*args):
    """Return True if all of the (non-None) elements are equal"""
    return set(filter(lambda x: x is not None, args))


def all_none(*args):
    """Return True if all arguments passed in are None."""
    return all(a is None for a in args)
