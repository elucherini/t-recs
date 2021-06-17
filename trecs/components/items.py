""" Class definition for Items in our recommender systems - can represent anything
    ranging from disease to consumer goods
"""
import trecs.matrix_ops as mo
from trecs.base import Component


class Items(Component):  # pylint: disable=too-many-ancestors
    """
    Items components in the system.

    Items are the objects with which users interact with models and with each other.

    It inherits from :class:`~.base_components.Component` and it contains all the
    attributes of that class. We support both numpy arrays or scipy sparse matrices
    being used for the item representation.

    The Items class does not specify any constraints on items. It is used
    internally in :class:`~models.recommender.BaseRecommender`.

    Parameters
    -------------

    item_attributes: array_like, optional
        Representation of items. It expects an array_like object. If None, the
        representation is generated randomly using a binomial distribution (see
        :class:`~base.base_components.Component` for details). At least one of
        `item_attributes` or `size` must be supplied.

    size: tuple, optional
        Size of the item representation. It expects a tuple. At least one of
        `item_attributes` or `size` must be supplied.

    verbose: bool, default False
        If True, it enables verbose mode.

    seed: int, optional
        Seed for underlying random generator.

    Attributes
    ------------

    Attributes from Component
        Inherited from :class:`~base.base_components.Component`

    name: str
        Name of the component

    """

    def __init__(self, item_attributes=None, size=None, verbose=False, seed=None, name="items"):
        self.name = name
        Component.__init__(
            self, current_state=item_attributes, size=size, verbose=verbose, seed=seed
        )

    def append_new_items(self, new_items):
        """
        Appends a set of new items (represented as some kind of matrix) to the current
        set of items. Assumes the new items have dimension :math:`|A|\\times|I_{new}|`,
        where :math:`I_{new}` indicates the number of new items to be appended.
        """
        self.current_state = mo.hstack([self.current_state, new_items])

    @property
    def num_attrs(self):
        """
        Shortcut getter method for the number of attributes of the items.
        """
        # rows = attributes, cols = items
        return self.current_state.shape[0]

    @property
    def num_items(self):
        """
        Shortcut getter method for the number of items.
        """
        # rows = attributes, cols = items
        return self.current_state.shape[1]


class PredictedItems(Items):  # pylint: disable=too-many-ancestors
    """
    This component represents the item attributes, as predicted by the
    recommender system. The only difference between the two is the intended
    purpose - all functionality is identical to the Items class
    and therefore PredictedItems inherits from it.

    Parameters
    -------------

    item_attributes: array_like, optional
        Representation of items. It expects an array_like object. If None, the
        representation is generated randomly using a binomial distribution (see
        :class:`~base.base_components.Component` for details). At least one of
        `item_attributes` or `size` must be supplied.

    size: tuple, optional
        Size of the item representation. It expects a tuple. At least one of
        `item_attributes` or `size` must be supplied.

    verbose: bool, default False
        If True, it enables verbose mode.

    seed: int, optional
        Seed for underlying random generator.

    Attributes
    ------------

    Attributes from Item
        Inherited by :class:`~.items.Items`

    name: str
        Name of the component

    """

    def __init__(self, item_attributes=None, size=None, verbose=False, seed=None):
        Items.__init__(
            self,
            item_attributes=item_attributes,
            size=size,
            verbose=verbose,
            seed=seed,
            name="predicted_items",
        )
