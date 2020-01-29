from .debug import VerboseMode
from .stats import Distribution
from .utils import normalize_matrix
import numpy as np

class Items(VerboseMode):
    def __init__(self, num_items=None, num_attributes=None, normalize=False,
        verbose=False, distribution=None):
        # Initialize verbose mode
        super().__init__(__name__.upper(), verbose)

        # default if distribution not specified or type not valid
        if not isinstance(distribution, Distribution):
            self.distribution = Distribution('norm', non_negative=True)
            self.log('Initialized default normal distribution with no size')
        else:
            self.distribution = distribution

        # Set up Items instance
        if not isinstance(num_items, int):
            self.num_items = None
            self.num_attributes = num_attributes
            self.item_attributes = None
            self.log('Initialized empty ActualUserScores instance')
            return

        self.num_items = num_items

        if not isinstance(num_attributes, int):
            # default num_attributes if not specified or not valid
            self.num_attributes = None
        else:
            self.num_attributes = num_attributes

        # Compute items with (|A|x|I|)
        self.item_attributes = self.compute_item_attributes(num_items=self.num_items,
                                        num_attributes=self.num_attributes,
                                        distribution=self.distribution, 
                                        normalize=False)

    def _compute_item_attributes(self, num_items, num_attributes, normalize=False):
        # Compute item attributes (|A|x|I|)
        assert(num_items is not None and num_attributes is not None)
        item_attributes = self.distribution.compute(size=(self.num_attributes, self.num_items))
        # Normalize
        if normalize:
            item_attributes = normalize_matrix(item_attributes, axis=1)
        return item_attributes

    def compute_item_attributes(self, num_items=1250, num_attributes=None, distribution=None, 
                                            normalize=False):
        # Error if num_items or num_attributes not valid
        if self.num_items is None and not isinstance(num_items, int):
            raise TypeError("num_items must be int or None")
        else:
            self.num_items = num_items
        # Default num_attributes if not specified or not valid (9/10th of num_items)
        if self.num_attributes is None and not isinstance(num_attributes, int):
            self.num_attributes = int(self.num_items - self.num_items / 10)
        else:
            self.num_attributes = num_attributes
        # Use specified distribution if specified, otherwise default to self.distribution
        if distribution is not None and isinstance(distribution, Distribution):
            self.distribution = distribution
        self.item_attributes = self._compute_item_attributes(self.num_attributes,
            self.num_items, normalize=normalize)
        return self.item_attributes


    def expand_items(self, num_new_items, normalize=False):
        new_indices = np.arange(self.num_items, self.num_items + num_new_items)
        self.num_items += num_new_items
        self.log('Successfully added %d new items' % num_new_items)
        new_item_attributes = self._compute_item_attributes(num_new_items, num_attributes,
            normalize=normalize)
        self.item_attributes = np.concatenate((self.item_attributes, new_item_attributes),
            axis=1)
        return new_indices


