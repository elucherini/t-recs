import numpy as np
import pytest
import test_helpers
from trecs.validate import validate_user_item_inputs


class TestValidate:
    def test_validate_good_user_item_inputs(self):
        num_users = 300
        num_items = 600
        users_hat = np.ones((300, 100))
        items_hat = np.ones((100, 600))
        users = np.ones((300, 200))
        items = np.ones((200, 600))
        default_users = 400
        default_items = 700
        default_attrs = 500
        num_attributes = 100

        # only pass in num users and num items
        # note that all other variables must be explicitly passed in as None
        assert validate_user_item_inputs(num_users, num_items, attributes_must_match=False) == (
            num_users,
            num_items,
        )

        # just pass in user and item representations
        num_users_inferred, num_items_inferred, num_attr_inferred = validate_user_item_inputs(
            users_hat=users_hat, items_hat=items_hat
        )
        assert num_users_inferred == num_users
        assert num_items_inferred == num_items
        assert num_attr_inferred == num_attributes

        # just pass in true users and items matrices
        num_users_inferred, num_items_inferred, num_attr_inferred = validate_user_item_inputs(
            users=users, items=items
        )
        assert num_users_inferred == num_users
        assert num_items_inferred == num_items
        assert (
            num_attr_inferred == None
        )  # attrs only applies to user / item hat, not the true users/items

        # pass in nothing but the default values
        num_users_inferred, num_items_inferred, num_attr_inferred = validate_user_item_inputs(
            default_num_users=default_users,
            default_num_items=default_items,
            default_num_attributes=default_attrs,
        )
        assert num_users_inferred == default_users
        assert num_items_inferred == default_items
        assert num_attr_inferred == default_attrs

        # pass in everything and verify that it is all consistent
        num_users_inferred, num_items_inferred, num_attr_inferred = validate_user_item_inputs(
            num_users,
            num_items,
            users_hat,
            items_hat,
            users,
            items,
            default_users,
            default_items,
            default_attrs,
            num_attributes,
        )
        assert num_users_inferred == num_users
        assert num_items_inferred == num_items
        assert num_attr_inferred == num_attributes

    def test_validate_bad_user_item_inputs(self):
        num_users = 300
        num_items = 600
        users_hat = np.ones((300, 100))
        items_hat = np.ones((100, 600))
        users = np.ones((300, 200))
        items = np.ones((200, 600))

        # users hat array doesn't contain the right number of users
        with pytest.raises(ValueError):
            validate_user_item_inputs(num_users, num_items, users_hat=np.ones((299, 100)))

        # users and items do not match on attributes
        with pytest.raises(ValueError):
            validate_user_item_inputs(
                num_users, num_items, users_hat=users_hat, items_hat=np.ones((99, 600))
            )

        # users and items do not match num_users and num_items
        with pytest.raises(ValueError):
            validate_user_item_inputs(100, num_items, users_hat=users_hat, items_hat=items_hat)

        with pytest.raises(ValueError):
            validate_user_item_inputs(num_users, 200, users_hat=users_hat, items_hat=items_hat)

        with pytest.raises(ValueError):
            validate_user_item_inputs(100, num_items, users=users, items=items)

        with pytest.raises(ValueError):
            validate_user_item_inputs(num_users, 200, users=users, items=items)

        # num_attributes does not match the actual number of attributes
        with pytest.raises(ValueError):
            validate_user_item_inputs(
                users_hat=users_hat,
                items_hat=items_hat,
                num_attributes=500,
            )

        # runs without error
        validate_user_item_inputs(
            users_hat=users_hat,
            items_hat=items_hat,
            num_attributes=500,
            attributes_must_match=False,
        )

        # true users and true items do not match the number of users and items
        # in the representation
        with pytest.raises(ValueError):
            validate_user_item_inputs(
                users_hat=np.ones((299, 100)),
                items_hat=items_hat,
                users=users,
                items=items,
            )

        with pytest.raises(ValueError):
            validate_user_item_inputs(
                users_hat=users_hat,
                items_hat=np.ones((100, 599)),
                users=users,
                items=items,
            )
