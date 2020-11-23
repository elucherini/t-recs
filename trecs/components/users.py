"""
Suite of classes related to users of the system, including predicted user-item
scores, predicted user profiles, actual user profiles, and a Users class (which
encapsulates some of these concepts)
"""
import numpy as np

from trecs.matrix_ops import contains_row, slerp, inner_product
from trecs.random import Generator
from .base_components import Component, BaseComponent


class PredictedScores(Component):  # pylint: disable=too-many-ancestors
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


class PredictedUserProfiles(Component):  # pylint: disable=too-many-ancestors
    """
    User profiles as predicted by the model. This class is a container
    compatible with Numpy operations and it does not make assumptions on the
    size of the representation.
    """

    def __init__(self, user_profiles=None, size=None, verbose=False, seed=None):
        self.name = "predicted_user_profiles"
        Component.__init__(self, current_state=user_profiles, size=size, verbose=verbose, seed=seed)


class ActualUserProfiles(Component):  # pylint: disable=too-many-ancestors
    """
    Real user profiles, unknown to the model. This class is a container
    compatible with Numpy operations and it does not make assumptions on the
    size of the representation.
    """

    def __init__(self, user_profiles=None, size=None, verbose=False, seed=None):
        self.name = "actual_user_profiles"
        Component.__init__(self, current_state=user_profiles, size=size, verbose=verbose, seed=seed)


class Users(BaseComponent):  # pylint: disable=too-many-ancestors
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

        drift: float (optional, default: 0)
            If greater than 0, user profiles will update dynamically as they
            interact with items, "drifting" towards the item attribute vectors
            they interact with. `drift` is a parameter between 0 and 1 that
            controls the degree of rotational drift. If `t=1`, then the user
            profile vector takes on the exact same direction as the attribute
            vector of the item they just interacted with. If 0, user profiles
            are generated once at initialization and never change.

        score_fn: callable
            Function that is used to calculate each user's scores for each
            candidate item. The score function should take as input
            user_profiles and item_attributes.

        verbose: bool (optional, default: False)
            If True, enables verbose mode. Disabled by default.

        seed: int, None (optional, default: None)
            Seed for random generator.

    Attributes
    ------------

        Attributes from BaseComponent
            Inherited by :class:`~trecs.components.base_components.BaseComponent`

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

        user_vector: :obj:`numpy.ndarray`
            A ```|U|``` array of user indices.

        score_fn: callable
            Function that is used to calculate each user's scores for each
            candidate item. The score function should take as input
            user_profiles and item_attributes.

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
        drift=0,
        score_fn=inner_product,
        verbose=False,
        seed=None,
    ):  # pylint: disable=too-many-arguments
        self.rng = Generator(seed=seed)
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
            row_zeros = np.zeros(size[1])  # one row vector of zeroes
            while actual_user_profiles is None or contains_row(actual_user_profiles, row_zeros):
                # generate matrix until no row is the zero vector
                actual_user_profiles = self.rng.normal(size=size)
        self.actual_user_profiles = ActualUserProfiles(np.asarray(actual_user_profiles))
        self.interact_with_items = interact_with_items
        self.drift = drift
        assert callable(score_fn)
        self.score_fn = score_fn  # function that dictates how scores will be generated
        self.actual_user_scores = actual_user_scores
        if num_users is not None:
            self.user_vector = np.arange(num_users, dtype=int)
        self.name = "actual_user_scores"
        BaseComponent.__init__(self, verbose=verbose, init_value=self.actual_user_scores)

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
            user_profiles and item_attributes.

        Raises
        --------

        TypeError
            If score_fn is not callable.
        """
        if not callable(score_fn):
            raise TypeError("score function must be callable")
        self.score_fn = score_fn

    def compute_user_scores(self, item_attributes):
        """
        Computes and stores the actual scores that users assign to items
        compatible with the system. Note that we expect the score_fn
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
            user_profiles=self.actual_user_profiles, item_attributes=item_attributes
        )
        if self.actual_user_scores is None:
            self.actual_user_scores = actual_scores
        else:
            self.actual_user_scores[:, :] = actual_scores

        self.store_state()

    def score_new_items(self, new_items):
        """
        Computes and stores the actual scores that users assign to any new
        items that enter the system. Note that we expect the score_fn
        attribute to be set to some callable function which takes item
        attributes and user profiles.

        Parameters
        ------------

        new_items: :obj:`array_like`
            A matrix representation of item attributes. Should be of dimension
            :math:`|A|\\times|I|`, where :math:`|I|` is the
            number of items and :math:`|A|` is the number of attributes.
        """
        new_scores = self.score_fn(
            user_profiles=self.actual_user_profiles, item_attributes=new_items
        )
        self.actual_user_scores = np.hstack([self.actual_user_scores, new_scores])
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

        .. todo::

            Raise exceptions

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

        items_shown: :obj:`numpy.ndarray`): A
            :math:`|U|\\times\\text{num_items_per_iter}` matrix with
            recommendations and new items.
        item_attributes: :obj:`numpy.ndarray`):
            A :math:`|A|\\times|I|` matrix with item attributes.

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
        items_shown = kwargs.pop("items_shown", None)
        item_attributes = kwargs.pop("item_attributes", None)
        if items_shown is None:
            raise ValueError("Items can't be None")
        reshaped_user_vector = self.user_vector.reshape((items_shown.shape[0], 1))
        user_interactions = self.actual_user_scores[reshaped_user_vector, items_shown]
        self.log("User scores for given items are:\n" + str(user_interactions))
        sorted_user_preferences = user_interactions.argsort()[:, -1]
        interactions = items_shown[self.user_vector, sorted_user_preferences]
        self.log("Users interact with the following items respectively:\n" + str(interactions))
        if self.drift > 0:
            if item_attributes is None:
                raise ValueError("Item attributes can't be None if user preferences are dynamic")
            # update user profiles based on the attributes of items they
            # interacted with
            interact_attrs = item_attributes.T[interactions, :]
            self.update_profiles(interact_attrs)
            # update user scores
            self.compute_user_scores(item_attributes)
        return interactions

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
        self.actual_user_profiles = slerp(
            self.actual_user_profiles, item_attributes, perc=self.drift
        )

    def store_state(self):
        """ Store the actual user scores in the state history """
        self.state_history.append(np.copy(self.actual_user_scores))


class DNUsers(Users):
    """
    Subclass of :class:`~components.users.Users` in which user agents perform
    choices in accordance with the Divisive Normalization model of choice
    from `Webb et al., 2020`_.

    .. _Webb et al., 2020: https://pubsonline.informs.org/doi/pdf/10.1287/mnsc.2019.3536

    Parameters
    -----------
        sigma: float
            Parameter for the DN model (see docstring). Default value is fitted
            parameter from Webb et al. (2020).

        omega: float
            Parameter for the DN model (see docstring). Default value is fitted
            parameter from Webb et al. (2020).

        beta: float
            Parameter for the DN model (see docstring). Default value is fitted
            parameter from Webb et al. (2020).
    """

    def __init__(
        self,
        actual_user_profiles=None,
        actual_user_scores=None,
        interact_with_items=None,
        size=None,
        num_users=None,
        drift=0,
        score_fn=inner_product,
        sigma=0.0,
        omega=0.2376,
        beta=0.9739,
        verbose=False,
        seed=None,
    ):  # pylint: disable=too-many-arguments
        Users.__init__(
            self,
            actual_user_profiles,
            actual_user_scores,
            interact_with_items,
            size,
            num_users,
            drift,
            score_fn,
            verbose,
            seed,
        )
        self.sigma = sigma
        self.omega = omega
        self.beta = beta

    def get_user_feedback(self, *args, **kwargs):
        """
        Generates user interactions at a given timestep, generally called by a
        model.

        Parameters
        ------------

        args, kwargs:
            Parameters needed by the model's train function.

        items_shown: :obj:`numpy.ndarray`): A
            :math:`|U|\\times\\text{num_items_per_iter}` matrix with
            recommendations and new items.

        item_attributes: :obj:`numpy.ndarray`):
            A :math:`|A|\\times|I|` matrix with item attributes.

        Returns
        ---------
            Array of interactions s.t. element :math:`interactions_{u(t)}` represents the
            index of the item selected by user `u` at time `t`. Shape: |U|

        Raises
        -------

        ValueError
            If :attr:`interact_with_items` is None and there is not `item`
            parameter.
        """
        if self.interact_with_items is not None:
            return self.interact_with_items(*args, **kwargs)
        items_shown = kwargs.pop("items_shown", None)
        item_attributes = kwargs.pop("item_attributes", None)
        if items_shown is None:
            raise ValueError("Items can't be None")
        reshaped_user_vector = self.user_vector.reshape((items_shown.shape[0], 1))
        interaction_scores = self.actual_user_scores[reshaped_user_vector, items_shown]

        self.log("User scores for given items are:\n" + str(interaction_scores))
        item_utilities = self.dn_utilities(interaction_scores)
        sorted_user_preferences = item_utilities.argsort()[:, -1]
        interactions = items_shown[self.user_vector, sorted_user_preferences]
        self.log("Users interact with the following items respectively:\n" + str(interactions))

        if self.drift > 0:
            if item_attributes is None:
                raise ValueError("Item attributes can't be None if user preferences are dynamic")
            # update user profiles based on the attributes of items they
            # interacted with
            interact_attrs = item_attributes.T[interactions, :]
            self.update_profiles(interact_attrs)
            # update user scores
            self.compute_user_scores(item_attributes)
        return interactions

    def normalize_values(self, user_item_scores):
        """
        Calculating the expression for :math:`z(\\textbf{v})` in the equation
        :math:`z(\\textbf{v})+\\mathbf{\\eta}`.

        Parameters
        -----------

        user_item_scores: :obj:`array_like`
            The element at index :math:`i,j` should represent user :math:`i`'s
            context-independent value for item :math:`j`.
            Dimension: :math:`|U|\\times|I|`

        Returns
        --------
            normed_values: :obj:`numpy.ndarray`
                Probabilities of a user making a particular choice. All probabilities
                for a given row will sum up to 1.
        """
        summed_norms = np.linalg.norm(user_item_scores, ord=self.beta, axis=1)
        denom = self.sigma + np.multiply(self.omega, summed_norms)
        return np.divide(user_item_scores.T, denom)  # now |I| x |U|

    def dn_utilities(self, user_item_scores):
        """
        Scores items according to divisive normalization. Note that the parameters
        / matrix operations we perform here are directly taken from
        https://github.com/UofT-Neuroecon-1/Normalization. For more information,
        see Webb, R., Glimcher, P. W., & Louie, K. (2020). The Normalization of
        Consumer Valuations: Context-Dependent Preferences from Neurobiological
        Constraints. Management Science.

        Note that the generalized DN model takes the following functional form:
        :math:`z_i(\\textbf{v})=\\frac{v_i}{\\sigma+\\omega(\\sum_n v_n^{\\beta})^{\\frac{1}{\\beta}}}`,
        where :math:`\\sigma, \\omega, \\beta` are all parameters that specify the
        exact choice model. After the original values :math:`\\textbf{v}` are
        transformed this way, the choice is determined by choosing the maximum
        value over :math:`z(\\textbf{v})+\\mathbf{\\eta}`, which in our case
        is generated by a multivariate normal distribution.

        Parameters
        -----------

        user_item_scores: :obj:`array_like`
            The element at index :math:`i,j` should represent user :math:`i`'s
            context-independent value for item :math:`j`.
            Dimension: :math:`|U|\\times|I|`

        Returns
        --------
            utility: :obj:`numpy.ndarray`
                Normalized & randomly perturbed utilities for different each
                pair of users and items in the recommendation set.
        """
        normed_values = self.normalize_values(user_item_scores)
        num_choices, num_users = normed_values.shape
        # in accordance with the DN model from Webb et al.,
        # the following covariance matrix has the structure
        # [ 1     0.5   ...   0.5   0.5 ]
        # [ 0.5    1    ...   0.5   0.5 ]
        # [ 0.5   0.5   ...    1    0.5 ]
        # [ 0.5   0.5   ...   0.5    1  ]
        mean = np.zeros(num_choices)
        cov = np.ones((num_choices, num_choices)) * 0.5
        cov[np.arange(num_choices), np.arange(num_choices)] = 1
        # generate |I| x |U| multivariate normal
        eps = self.rng.multivariate_normal(mean, cov, size=num_users).T
        utility = normed_values + eps
        # transform so |U| x |I|
        return utility.T
