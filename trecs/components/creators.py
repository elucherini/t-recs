"""
Suite of classes related to content creators, including predicted user-item
scores, predicted user profiles, actual creator profiles, and a Creators class (which
encapsulates some of these concepts)
"""
import numpy as np

from trecs.matrix_ops import contains_row
from trecs.random import Generator
from .base_components import BaseComponent


class Creators(BaseComponent):  # pylint: disable=too-many-ancestors
    """
    Class representing content creators in the system.

    Each content creator is represented with a single vector that governs
    the kinds of content each creator produces. All creator profiles can be
    represented with a numpy ndarray of size
    `(number_of_creators, number_of_attributes)`.

    This class inherits from :class:`~components.base_components.BaseComponent`.

    Parameters
    ------------

        actual_creator_profiles: array_like or None (optional, default: None)
            Representation of the creator's attribute profiles.

        creation_probability: float, (optional, default: 0.5)
            The probability that any given creator produces a new item at a
            timestep.

        size: tuple, None (optional, default: None)
            Size of the user representation. It expects a tuple. If None,
            it is chosen randomly.

        verbose: bool (optional, default: False)
            If True, enables verbose mode. Disabled by default.

        seed: int, None (optional, default: None)
            Seed for random generator.

    Attributes
    ------------

        Attributes from BaseComponent
            Inherited by :class:`~trecs.components.base_components.BaseComponent`

        actual_creator_profiles: :obj:`numpy.ndarray`
            A matrix representing the *real* similarity between each item and
            attribute.

        create_new_items: callable
            A function that defines user behaviors when interacting with items.
            If None, users follow the behavior in :meth:`generate_new_items()`.

    Raises
    --------

        TypeError
            If parameters are of the wrong type.

        ValueError
            If both actual_creator_profiles and size are None.
    """

    def __init__(
        self,
        actual_creator_profiles=None,
        creation_probability=0.5,
        size=None,
        verbose=False,
        seed=None,
    ):  # pylint: disable=too-many-arguments
        # general input checks
        if actual_creator_profiles is not None:
            if not isinstance(actual_creator_profiles, (list, np.ndarray)):
                raise TypeError("actual_creator_profiles must be a list or numpy.ndarray")
        if actual_creator_profiles is None and size is None:
            raise ValueError("actual_creator_profiles and size can't both be None")
        if actual_creator_profiles is None and not isinstance(size, tuple):
            raise TypeError("size must be a tuple, is %s" % type(size))
        if actual_creator_profiles is None and size is not None:
            row_zeros = np.zeros(size[1])  # one row vector of zeroes
            while actual_creator_profiles is None or contains_row(
                actual_creator_profiles, row_zeros
            ):
                # generate matrix until no row is the zero vector
                actual_creator_profiles = Generator(seed=seed).uniform(size=size)
        if creation_probability > 1 or creation_probability < 0:
            raise ValueError("Creation probability cannot be less than 0 or greater than 1")
        self.actual_creator_profiles = np.asarray(actual_creator_profiles)
        self.creation_probability = creation_probability
        self.name = "actual_creator_profiles"
        BaseComponent.__init__(
            self, verbose=verbose, init_value=self.actual_creator_profiles, seed=seed
        )

    def generate_items(self):
        """
        Generates new items. Each creator probabilistically creates a new item.
        Item attributes are generated using each creator's profile
        as a series of Bernoulli random variables. Therefore, item attributes
        will be binarized. To change this behavior, simply write a custom
        class that overwrites this method.

        Returns
        ---------
            A numpy matrix of dimension :math:`|A|\\times|I_n|`, where
            :math:`|I_n|` represents the number of new items, and :math:`|A|`
            represents the number of attributes for each item.
        """
        # Generate mask by tossing coin for each creator to determine who is releasing content
        # This should result in a _binary_ matrix of size (num_creators,)
        if (self.actual_creator_profiles < 0).any() or (self.actual_creator_profiles > 1).any():
            raise ValueError("Creator profile attributes must be between zero and one.")
        creator_mask = Generator(seed=self.seed).binomial(
            1, self.creation_probability, self.actual_creator_profiles.shape[0]
        )
        chosen_profiles = self.actual_creator_profiles[creator_mask == 1, :]
        # for each creator that will add new items, generate Bernoulli trial
        # for item attributes
        items = Generator(seed=self.seed).binomial(
            1, chosen_profiles.reshape(-1), chosen_profiles.size
        )
        return items.reshape(chosen_profiles.shape).T

    def update_profiles(self, interactions, items):
        """
        This method can be implemented by child classes to update the
        creator profiles over time.

        Parameters
        -----------

            interactions: numpy.ndarray or list
                A matrix where row `i` corresponds to the attribute vector
                that user `i` interacted with.
        """
        # this can be overwritten by a custom creator class

    def store_state(self):
        """ Store the actual creator profiles in the state history """
        self.state_history.append(np.copy(self.actual_creator_profiles))
