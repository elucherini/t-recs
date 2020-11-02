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
            Representation of the real user profiles.

        actual_creator_scores: array_like or None (optional, default: None)
            Representation of the real scores that users assign to items.

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

        actual_creator_scores: :obj:`numpy.ndarray`
             A ```|U|x|I|``` matrix representing the *real* scores assigned by
             each user to each item, where ```|U|``` is the number of users and
             ```|I|``` is the number of items in the system. Item `[u, i]` is
             the score assigned by user `u` to item `i`.

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
        actual_creator_scores=None,
        create_new_items=None,
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
        if actual_creator_scores is not None:
            if not isinstance(actual_creator_scores, (list, np.ndarray)):
                raise TypeError("actual_creator_profiles must be a list or numpy.ndarray")
        if actual_creator_profiles is None and size is not None:
            row_zeros = np.zeros(size[1])  # one row vector of zeroes
            while actual_creator_profiles is None or contains_row(
                actual_creator_profiles, row_zeros
            ):
                # generate matrix until no row is the zero vector
                actual_creator_profiles = Generator(seed=seed).normal(size=size)
        self.actual_creator_profiles = ActualCreatorProfiles(np.asarray(actual_creator_profiles))
        self.create_new_items = create_new_items
        self.drift = drift
        self.score_fn = None  # function that dictates how items will be scored
        # this will be initialized by the system
        self.actual_creator_scores = actual_creator_scores
        if num_creators is not None:
            self.creator_vector = np.arange(num_creators, dtype=int)
        self.name = "actual_creator_scores"
        BaseComponent.__init__(
            self, verbose=verbose, init_value=self.actual_creator_scores, seed=seed
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

    def compute_creator_scores(self, item_attributes):
        """
        Computes and stores the actual scores that users assign to items
        compatible with the system. Note that we expect that the score_fn
        attribute to be set to some callable function which takes item
        attributes and user profiles.

        Parameters
        ------------

        item_attributes: :obj:`array_like`
            A matrix representation of item attributes.
        """
        if not callable(self.score_fn):
            raise TypeError("score function must be callable")
        actual_scores = self.score_fn(
            creator_profiles=self.actual_creator_profiles, item_attributes=item_attributes
        )
        if self.actual_creator_scores is None:
            self.actual_creator_scores = actual_scores
        else:
            self.actual_creator_scores[:, :] = actual_scores

        self.store_state()

    def get_actual_creator_scores(self, creator=None):
        """
        Returns an array of actual user scores.

        Parameters
        -----------

            creator: int or numpy.ndarray or list (optional, default: None)
                Specifies the user index (or indices) for which to return the
                actual user scores. If None, the function returns the whole
                matrix.

        Returns
        --------

            An array of actual user scores for each item.

        Todo
        -------

        * Raise exceptions

        """
        if creator is None:
            return self.actual_creator_scores
        else:
            return self.actual_creator_scores[creator, :]

    def generate_new_items(self, *args, **kwargs):
        """
        Generates user interactions at a given timestep, generally called by a
        model.

        Parameters
        ------------

        args, kwargs:
            Parameters needed by the model's train function.

            items_shown: :obj:`numpy.ndarray`): A |U|x|num_items_per_iter| matrix with
            recommendations and new items.
            item_attributes: :obj:`numpy.ndarray`): A |A|x|I| matrix with
            item attributes.

        Returns
        ---------
            Array of interactions s.t. element interactions_u(t) represents the
            index of the item selected by user u at time t. Shape: |U|

        Raises
        -------

        ValueError
            If :attr:`create_new_items` is None and there is not `item`
            parameter.
        """
        if self.create_new_items is not None:
            return self.create_new_items(*args, **kwargs)
        # Generate mask by tossing coin for each creator to determine who is releasing content
        # This should result in a _binary_ matrix of size (num_creators,)
        creator_mask = Generator(seed=self.seed).binomial(
            n=1,
            p=[1 / 2] * self.actual_creator_profiles.shape[0],
        )
        # I want to mask self.actual_creator_profiles with the results from creator_mask
        # First I need to repeat each element of creator_mask as many times as actual_creator_profiles's columns
        # and then reshape to obtain an array of the same size as actual_creator_profiles
        # TODO: check that the following is correct
        creator_mask = np.repeat(creator_mask, self.actual_creator_profiles.shape[1]).reshape(
            self.actual_creator_profiles.shape
        )
        masked_profiles = np.ma.masked_array(self.actual_creator_profiles, creator_mask)
        # for each creator that will add new items, generate multivariate_normal with each profile
        # TODO: check that the following is correct
        items = Generator(seed=self.seed).multivariate_normal(
            masked_profiles.ravel(), np.eye(masked_profiles)
        )
        # TODO: check reshape
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
        self.state_history.append(np.copy(self.actual_creator_scores))
