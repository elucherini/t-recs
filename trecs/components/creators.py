"""
Suite of classes related to content creators, including predicted user-item
scores, predicted user profiles, actual creator profiles, and a Creators class (which
encapsulates some of these concepts)
"""
import numpy as np

from trecs.matrix_ops import contains_row, slerp
from trecs.random import Generator
from .base_components import Component, BaseComponent


class ActualCreatorProfiles(Component):  # pylint: disable=too-many-ancestors
    """
    Real user profiles, unknown to the model. This class is a container
    compatible with Numpy operations and it does not make assumptions on the
    size of the representation.
    """

    def __init__(self, user_profiles=None, size=None, verbose=False, seed=None):
        self.name = "actual_user_profiles"
        Component.__init__(self, current_state=user_profiles, size=size, verbose=verbose, seed=seed)


class Creators(BaseComponent):  # pylint: disable=too-many-ancestors
    """
    Class representing users in the system.

    This class contains the real user preferences, which are unknown to the
    system, and the behavior of users when interacting with items.

    In general, users are represented with single *array_like* objects that
    contain all the users' preferences and characteristics. For example, real
    user preferences can be represented by a Numpy ndarray of size
    `(number_of_creators, number_of_items)` where element `[u,i]` is the score
    assigned by user u to item i.

    Models determine the size constraints of objects representing users.
    Requirements vary across models and, unless specified, this class does not
    make assumptions on the real user components.

    This class inherits from :class:`~components.base_components.BaseComponent`.

    Parameters
    ------------

        actual_creator_profiles: array_like or None (optional, default: None)
            Representation of the creator's attribute profiles.

        create_new_items: callable or None (optional, default: None)
            Function that specifies the behavior of users when interacting with
            items. If None, users follow the behavior specified in
            :meth:`generate_new_items()`.

        num_creators: int or None, (optional, default: None)
            The number of users in the system.

        size: tuple, None (optional, default: None)
            Size of the user representation. It expects a tuple. If None,
            it is chosen randomly.

        drift: float (optional, default: 0)
            If greater than 0, user profiles will update dynamically as they
            interact with items, "drifting" towards the item attribute vectors
            they interact with. `drift` is a parameter between 0 and 1 that
            controls the degree of rotational drift. If `t=1`, then the user
            profile vector takes on the exact same direction as the attribute
            vector of the item they just interacted with. If 0, user profiles
            are generated once at initialization and never change.

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

        creator_vector: :obj:`numpy.ndarray`
            A ```|U|``` array of user indices.

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
        create_new_items=None,
        creation_probability=0.5,
        size=None,
        num_creators=None,
        drift=0,
        verbose=False,
        seed=None,
    ):  # pylint: disable=too-many-arguments
        # general input checks
        if actual_creator_profiles is not None:
            if not isinstance(actual_creator_profiles, (list, np.ndarray)):
                raise TypeError("actual_creator_profiles must be a list or numpy.ndarray")
        if create_new_items is not None and not callable(create_new_items):
            raise TypeError("create_new_items must be callable")
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
        self.actual_creator_profiles = ActualCreatorProfiles(np.asarray(actual_creator_profiles))
        self.create_new_items = create_new_items
        self.creation_probability = creation_probability
        self.drift = drift
        self.score_fn = None  # function that dictates how items will be scored
        if num_creators is not None:
            self.creator_vector = np.arange(num_creators, dtype=int)
        self.name = "actual_creator_profiles"
        BaseComponent.__init__(
            self, verbose=verbose, init_value=self.actual_creator_profiles, seed=seed
        )

    def set_score_function(self, score_fn):
        """Users "score" items before "deciding" which item to interact with.
            This function makes it possible to set an arbitrary function as the
            score function.

        Parameters
        ------------

        score_fn: callable
            Function that is used to calculate each user's scores for each
            candidate item. Note that this function can be the same function
            used by the recommender system to generate its predictions for
            user-item scores. The score function should take as input
            creator_profiles and item_attributes.

        Raises
        --------

        TypeError
            If score_fn is not callable.
        """
        if not callable(score_fn):
            raise TypeError("score function must be callable")
        self.score_fn = score_fn


    def generate_new_items(self):
        """
        Generates new items. Each creator probabilistically creates a new item.
        Item attributes are generated using each creator's profile
        as a series of Bernoulli random variables. Therefore, item attributes
        will be binarized.

        Returns
        ---------
            A numpy matrix of dimension :math:`|I_n|\times|A|`, where
            :math:`|I_n|` represents the number of new items, and :math:`|A|`
            represents the number of attributes for each item.
        """
        if self.create_new_items is not None:
            return self.create_new_items(self.actual_creator_profiles)
        # Generate mask by tossing coin for each creator to determine who is releasing content
        # This should result in a _binary_ matrix of size (num_creators,)
        creator_mask = Generator(seed=self.seed).binomial(
            1,
            self.creation_probability,
            self.actual_creator_profiles.shape[0]
        )
        chosen_profiles = self.actual_creator_profiles[creator_mask == 1, :]
        # for each creator that will add new items, generate Bernoulli trial
        # for item attributes
        items = Generator(seed=self.seed).binomial(
            1, chosen_profiles.reshape(-1), chosen_profiles.size
        )
        return items.reshape(self.actual_creator_profiles.shape[0], -1)

    def update_profiles(self, item_attributes):
        """In the case of dynamic user profiles, we update the user's actual
        profiles with new values as each user profile "drifts" towards
        items that they consume.

        Parameters
        -----------

            interactions: numpy.ndarray or list
                A matrix where row `i` corresponds to the attribute vector
                that user `i` interacted with.
        """
        # we make no assumptions about whether the user profiles or item
        # attributes vectors are normalized
        self.actual_creator_profiles = slerp(
            self.actual_creator_profiles, item_attributes, perc=self.drift
        )

    def store_state(self):
        """ Store the actual user scores in the state history """
        self.state_history.append(np.copy(self.actual_creator_profiles))
