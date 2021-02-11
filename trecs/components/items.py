""" Class definition for Items in our recommender systems - can represent anything
    ranging from disease to consumer goods
"""
import trecs.matrix_ops as mo
from trecs.base import Component


class Items(Component):  # pylint: disable=too-many-ancestors
    """
    Items components in the system.

    Items are the objects with which users interact with models and with each other.

    This class is essentially a container of Numpy's ndarray. Therefore, it
    supports all the operations supported by ndarray. It inherits from
    :class:`~.base_components.Component` and it contains all the attributes of
    that class.

    The Items class does not specify any constraints on items. It is used
    internally in :class:`~models.recommender.BaseRecommender`.

    Parameters
    -------------

    item_attributes: array_like, None (optional, default: None)
        Representation of items. It expects an array_like object. If None, the
        representation is generated randomly.

    size: tuple, None (optional, default: None)
        Size of the item representation. It expects a tuple. If None, it is
        chosen randomly.

    verbose: bool (optional, default: False)
        If True, it enables verbose mode.

    seed: int, None (optional, default: None)
        Seed for underlying random generator.

    Attributes
    ------------

    Attributes from Component
        Inherited by :class:`~.base_components.Component`

    name: str
        Name of the component

    """

    def __init__(self, item_attributes=None, size=None, verbose=False, seed=None):
        self.name = "items"
        Component.__init__(
            self, current_state=item_attributes, size=size, verbose=verbose, seed=seed
        )

    def append_items(self, new_items):
        self.current_state = mo.hstack([self.current_state, new_items])

    @property
    def num_attr(self):
        # rows = attributes, cols = items
        return self.current_state.shape[0]

    @property
    def num_items(self):
        # rows = attributes, cols = items
        return self.current_state.shape[1]


class PredictedItems(Component):  # pylint: disable=too-many-ancestors
    """
    TODO: write docstring
    """

    def __init__(self, item_attributes=None, size=None, verbose=False, seed=None):
        self.name = "predicted_items"
        Component.__init__(
            self, current_state=item_attributes, size=size, verbose=verbose, seed=seed
        )

    def append_items(self, new_items):
        self.current_state = mo.hstack([self.current_state, new_items])

    @property
    def num_attr(self):
        # rows = attributes, cols = items
        return self.current_state.shape[0]

    @property
    def num_items(self):
        # rows = attributes, cols = items
        return self.current_state.shape[1]
