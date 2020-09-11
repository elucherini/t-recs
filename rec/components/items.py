from rec.utils import VerboseMode, normalize_matrix
from rec.random import Generator
from .base_components import Component
import numpy as np


class Items(Component):
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


if __name__ == "__main__":
    a = Items([1, 2, 3], verbose=True)
    print(a)
    print(type(a))
    print(a + 2)
    print(a.verbose)
