"""Set of methods specifically designed to validate user inputs"""
from trecs.components import Users
from trecs.utils import check_consistency


def validate_user_item_inputs(  # pylint: disable=too-many-arguments
    num_users=None,
    num_items=None,
    users_hat=None,
    items_hat=None,
    users=None,
    items=None,
    default_num_users=None,
    default_num_items=None,
    default_num_attributes=None,
    num_attributes=None,
    user_item_scores=None,
    attributes_must_match=True,
):
    """ Mostly a wrapper around `check_consistency`. The reason we
    have this method is that this method "accepts" different classes
    for its arguments (i.e., an instance of the `Users` class for the
    `users` argument), whereas `check_consistency` expects all arguments
    to be numpy arrays.

    Parameters
    -----------

    num_users: int or None (optional, default: None)
        An integer representing the number of users in the system

    num_items: int or None (optional, default: None)
        An integer representing the number of items in the system

    users_hat: :obj:`numpy.ndarray` or None (optional, default: None)
        A 2D matrix whose first dimension should be equal to the number of
        users in the system. Typically this matrix refers to the system's
        internal representation of user profiles, not the "true" underlying
        user profiles, which are unknown to the system.

    items_hat: :obj:`numpy.ndarray` or None (optional, default: None)
        A 2D matrix whose second dimension should be equal to the number of
        items in the system. Typically this matrix refers to the system's
        internal representation of item attributes, not the "true" underlying
        item attributes, which are unknown to the system.

    users: :obj:`numpy.ndarray` or :class:`~components.users.Users` \
            or None (optional, default: None)
        A 2D matrix whose first dimension should be equal to the number of
        users in the system. This is the "true" underlying user profile
        matrix.

    items: :obj:`numpy.ndarray` or None (optional, default: None)
        A 2D matrix whose second dimension should be equal to the number of
        items in the system. This is the "true" underlying item attribute
        matrix.

    default_num_users: int or None (optional, default: None)
        If the number of users is not specified anywhere in the inputs, we return
        this value as the number of users to be returned.

    default_num_items: int or None (optional, default: None)
        If the number of items is not specified anywhere in the inputs, we return
        this value as the number of items to be returned.

    default_num_attributes: int or None (optional, default: None)
        If the number of attributes in the item/user representations is not
        specified or cannot be inferred, this is the default number
        of attributes that should be used. (This applies only to users_hat
        and items_hat.)

    num_attributes: int or None (optional, default: None)
        Check that the number of attributes per user & per item are equal to
        this specified number. (This applies only to users_hat and items_hat.)

    user_item_scores: :obj:`numpy.ndarray` or None (optional, default: None)
        A 2D matrix whose first dimension is the number of users in the system
        and whose second dimension is the number of items in the system.

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
    if isinstance(users, Users):  # assume member of Users class
        if user_item_scores is not None and users.actual_user_scores is not None:
            if user_item_scores != users.actual_user_scores:
                raise AssertionError(
                    "Provided user-item scores not equal to user-item scores in Users object"
                )
        user_item_scores = users.actual_user_scores
        users = users.actual_user_profiles

    return check_consistency(
        num_users=num_users,
        num_items=num_items,
        users_hat=users_hat,
        items_hat=items_hat,
        users=users,
        items=items,
        default_num_users=default_num_users,
        default_num_items=default_num_items,
        default_num_attributes=default_num_attributes,
        num_attributes=num_attributes,
        user_item_scores=user_item_scores,
        attributes_must_match=attributes_must_match,
    )
