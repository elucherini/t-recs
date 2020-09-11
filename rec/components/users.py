import numpy as np

from rec.utils import VerboseMode, normalize_matrix
from rec.random import Generator
from .base_components import Component, BaseComponent


class PredictedScores(Component):
    """
    User scores about items generated by the model. This class is a container
    compatible with Numpy operations and it does not make assumptions on the
    size of the representation.
    """

    def __init__(self, predicted_scores=None, verbose=False):
        self.name = "predicted_user_scores"
        Component.__init__(
            self, current_state=predicted_scores, size=None, verbose=verbose, seed=None
        )


class PredictedUserProfiles(Component):
    """
    User profiles as predicted by the model. This class is a container
    compatible with Numpy operations and it does not make assumptions on the
    size of the representation.
    """

    def __init__(self, user_profiles=None, size=None, verbose=False, seed=None):
        self.name = "predicted_user_profiles"
        Component.__init__(
            self, current_state=user_profiles, size=size, verbose=verbose, seed=seed
        )


class ActualUserProfiles(Component):
    """
    Real user profiles, unknown to the model. This class is a container
    compatible with Numpy operations and it does not make assumptions on the
    size of the representation.
    """

    def __init__(self, user_profiles=None, size=None, verbose=False, seed=None):
        self.name = "actual_user_profiles"
        Component.__init__(
            self, current_state=user_profiles, size=size, verbose=verbose, seed=seed
        )


class Users(BaseComponent):
    """
    Class representing users in the system.

    This class contains the real user preferences, which are unknown to the
    system, and the behavior of users when interacting with items.

    In general, users are represented with single *array_like* objects that
    contain all the users' preferences and characteristics. For example, real
    user preferences can be represented by a Numpy ndarray of size
    `(number_of_users, number_of_items)` where element `[u,i]` is the score
    assigned by user u to item i.

    Models determine the size constraints of objects representing users.
    Requirements vary across models and, unless specified, this class does not
    make assumptions on the real user components.

    This class inherits from :class:`~components.base_components.BaseComponent`.

    Parameters
    ------------

        actual_user_profiles: array_like or None (optional, default: None)
            Representation of the real user profiles.

        actual_user_scores: array_like or None (optional, default: None)
            Representation of the real scores that users assign to items.

        interact_with_items: callable or None (optional, default: None)
            Function that specifies the behavior of users when interacting with
            items. If None, users follow the behavior specified in
            :meth:`get_user_feedback()`.

        num_users: int or None, (optional, default: None)
            The number of users in the system.

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
            Inherited by :class:`~rec.components.base_components.BaseComponent`

        actual_user_profiles: :obj:`numpy.ndarray`
            A matrix representing the *real* similarity between each item and
            attribute.

        actual_user_scores: :obj:`numpy.ndarray`
             A ```|U|x|I|``` matrix representing the *real* scores assigned by
             each user to each item, where ```|U|``` is the number of users and
             ```|I|``` is the number of items in the system. Item `[u, i]` is
             the score assigned by user `u` to item `i`.

        interact_with_items: callable
            A function that defines user behaviors when interacting with items.
            If None, users follow the behavior in :meth:`get_user_feedback()`.

        _user_vector: **private** :obj:`numpy.ndarray`
            A ```|U|``` array of user indices, used internally.

    Raises
    --------

        TypeError
            If parameters are of the wrong type.

        ValueError
            If both actual_user_profiles and size are None.
    """

    def __init__(
        self,
        actual_user_profiles=None,
        actual_user_scores=None,
        interact_with_items=None,
        size=None,
        num_users=None,
        verbose=False,
        seed=None,
    ):
        # general input checks
        if actual_user_profiles is not None:
            if not isinstance(actual_user_profiles, (list, np.ndarray)):
                raise TypeError("actual_user_profiles must be a list or numpy.ndarray")
        if interact_with_items is not None and not callable(interact_with_items):
            raise TypeError("interact_with_items must be callable")
        if actual_user_profiles is None and size is None:
            raise ValueError("actual_user_profiles and size can't both be None")
        if actual_user_profiles is None and not isinstance(size, tuple):
            raise TypeError("size must be a tuple, is %s" % type(size))
        if actual_user_scores is not None:
            if not isinstance(actual_user_scores, (list, np.ndarray)):
                raise TypeError("actual_user_profiles must be a list or numpy.ndarray")
        if actual_user_profiles is None and size is not None:
            actual_user_profiles = Generator(seed=seed).normal(size=size)
        self.actual_user_profiles = ActualUserProfiles(np.asarray(actual_user_profiles))
        self.interact_with_items = interact_with_items
        # this will be initialized by the system
        self.actual_user_scores = None
        if num_users is not None:
            self._user_vector = np.arange(num_users, dtype=int)
        self.name = "actual_user_scores"
        BaseComponent.__init__(
            self, verbose=verbose, init_value=self.actual_user_scores
        )

    def compute_user_scores(self, train_function, *args, **kwargs):
        """
        Computes and stores the actual scores that users assign to items
        compatible with the system. It does so by using the model's train function.

        Parameters
        ------------

        train_function: callable
            Function that is used to train the model. Since training the model
            corresponds to generating user scores starting from user profiles,
            as predicted by the model, the same function can be used to compute
            the real scores using the real user preferences.

        args, kwargs:
            Parameters needed by the model's train function.

        Raises
        --------

        TypeError
            If train_function is not callable.
        """
        if not callable(train_function):
            raise TypeError("train_function must be callable")
        actual_scores = train_function(
            user_profiles=self.actual_user_profiles, *args, **kwargs
        )
        if self.actual_user_scores is None:
            self.actual_user_scores = actual_scores
        else:
            self.actual_user_scores[:, :] = actual_scores

        self.store_state()

    def get_actual_user_scores(self, user=None):
        """
        Returns an array of actual user scores.

        Parameters
        -----------

            user: int or numpy.ndarray or list (optional, default: None)
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
        if user is None:
            return self.actual_user_scores
        else:
            return self.actual_user_scores[user, :]

    def get_user_feedback(self, *args, **kwargs):
        """
        Generates user interactions at a given timestep, generally called by a
        model.

        Parameters
        ------------

        args, kwargs:
            Parameters needed by the model's train function.

            items: :obj:`numpy.ndarray`): A |U|x|num_items_per_iter| matrix with
            recommendations and new items.

        Returns
        ---------
            Array of interactions s.t. element interactions_u(t) represents the
            index of the item selected by user u at time t. Shape: |U|

        Raises
        -------

        ValueError
            If :attr:`interact_with_items` is None and there is not `item`
            parameter.
        """
        if self.interact_with_items is not None:
            return self.interact_with_items(*args, **kwargs)
        items = kwargs.pop("items", None)
        if items is None:
            raise ValueError("Items can't be None")
        reshaped_user_vector = self._user_vector.reshape((items.shape[0], 1))
        user_interactions = self.actual_user_scores[reshaped_user_vector, items]
        self.log("User scores for given items are:\n" + str(user_interactions))
        sorted_user_preferences = user_interactions.argsort()[:, ::-1][:, 0]
        interactions = items[self._user_vector, sorted_user_preferences]
        self.log(
            "Users interact with the following items respectively:\n"
            + str(interactions)
        )
        return interactions

    def store_state(self):
        self.state_history.append(np.copy(self.actual_user_scores))
