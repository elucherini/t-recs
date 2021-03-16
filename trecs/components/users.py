"""
Suite of classes related to users of the system, including predicted user-item
scores, predicted user profiles, actual user profiles, and a Users class (which
encapsulates some of these concepts)
"""
import numpy as np
import scipy.sparse as sp

import trecs.matrix_ops as mo
from trecs.random import Generator
from trecs.utils import check_consistency
from trecs.base import Component, BaseComponent


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

    def filter_by_index(self, item_indices):
        """
        Return a subset of the predicted scores, filtered by the indices
        of valid items.

        Parameters
        -------------

        item_indices: :obj:`numpy.ndarray` or `scipy.sparse.spmatrix`
            A matrix with |U| rows that specifies the indices of items
            requested for each user.

        """
        if item_indices.shape[0] != self.current_state.shape[0]:
            error_msg = "Number of users does not match between score matrix and item index matrix"
            raise ValueError(error_msg)
        # generates row matrix like the following:
        # [0, 0, 0, ..., 0]
        # [1, 1, 1, ..., 1]
        # [     ...       ]
        # [n, n, n, ..., n]
        num_users = item_indices.shape[0]
        row = np.repeat(np.arange(num_users), item_indices.shape[1]).reshape((num_users, -1))
        # for now, we have to keep the score matrix a dense array because scipy
        # sparse has no equivalent of argsort
        # TODO: look into potential solutions using things like Numba to maintain
        # speed?
        # https://stackoverflow.com/questions/31790819/scipy-sparse-csr-matrix-how-to-get-top-ten-values-and-indices
        return mo.to_dense(self.current_state)[row, item_indices]

    def append_new_scores(self, new_scores):
        """
        Appends a set of scores for new items to the current set of scores.

        Parameters
        -------------

        new_scores: :obj:`numpy.ndarray` or `scipy.sparse.spmatrix`
            Matrix of new scores with dimension :math:`|U|\\times|I_{new}|`,
            where :math:`I_{new}` indicates the number of new items whose scores
            are being to be appended.
        """
        self.current_state = mo.hstack([self.current_state, new_scores])


class PredictedUserProfiles(Component):  # pylint: disable=too-many-ancestors
    """
    User profiles as predicted by the model. This class is a container
    compatible with Numpy operations and it does not make assumptions on the
    size of the representation.
    """

    def __init__(self, user_profiles=None, size=None, verbose=False, seed=None):
        self.name = "predicted_users"
        Component.__init__(self, current_state=user_profiles, size=size, verbose=verbose, seed=seed)

    @property
    def num_users(self):
        """
        Shortcut getter method for the number of users.
        """
        return self.current_state.shape[0]

    @property
    def num_attrs(self):
        """
        Shortcut getter method for the number of attributes in each user profile.
        """
        return self.current_state.shape[1]


class ActualUserProfiles(Component):  # pylint: disable=too-many-ancestors
    """
    Real user profiles, unknown to the model. This class is a container
    compatible with Numpy operations and it does not make assumptions on the
    size of the representation.
    """

    def __init__(self, user_profiles=None, size=None, verbose=False, seed=None):
        self.name = "actual_user_profiles"
        Component.__init__(self, current_state=user_profiles, size=size, verbose=verbose, seed=seed)


class ActualUserScores(Component):  # pylint: disable=too-many-ancestors
    """
    Real matrix of user-item scores, unknown to the model.
    """

    def __init__(self, user_profiles=None, size=None, verbose=False, seed=None):
        self.name = "actual_user_scores"
        if user_profiles is not None:
            num_users, num_items = user_profiles.shape
            self.user_rows = np.repeat(np.arange(num_users), num_items).reshape((num_users, -1))
        else:
            self.user_rows = None
        Component.__init__(self, current_state=user_profiles, size=size, verbose=verbose, seed=seed)

    def get_item_scores(self, items_shown):
        """
        Return the user scores for the items shown, in the correct
        order specified.
        """
        if self.user_rows is None or self.user_rows.shape != self.current_state.shape:
            num_users, num_items = self.current_state.shape
            self.user_rows = np.repeat(np.arange(num_users), num_items).reshape((num_users, -1))
        num_items = items_shown.shape[1]
        return self.current_state[self.user_rows[:, :num_items], items_shown]

    def set_item_scores_to_value(self, item_indices, value):
        """
        Set scores for the specified user-item indices to the determined
        value.

        Parameters
        -----------
            item_indices: :obj:`numpy.ndarray` or `scipy.sparse.spmatrix`
                A matrix with |U| rows that specifies the indices of items
                requested for each user.

            value: float
                Single value with which to replace scores.
        """
        if self.user_rows is None or self.user_rows.shape != self.current_state.shape:
            num_users, num_items = self.current_state.shape
            self.user_rows = np.repeat(np.arange(num_users), num_items).reshape((num_users, -1))
        num_items = item_indices.shape[1]
        self.current_state[self.user_rows[:, :num_items], item_indices] = value

    def append_new_scores(self, new_scores):
        """
        Appends a set of scores for new items to the current set of scores.

        Parameters
        -------------

        new_scores: :obj:`numpy.ndarray` or `scipy.sparse.spmatrix`
            Matrix of new scores with dimension :math:`|U|\\times|I_{new}|`,
            where :math:`I_{new}` indicates the number of new items whose scores
            are being to be appended.
        """
        self.current_state = mo.hstack([self.current_state, new_scores])
        # update user rows matrix
        num_users, num_items = self.current_state.shape
        self.user_rows = np.repeat(np.arange(num_users), num_items).reshape((num_users, -1))

    @property
    def num_users(self):
        """
        Shortcut getter method for the number of users.
        """
        # rows = users, cols = items
        return self.current_state.shape[0]

    @property
    def num_items(self):
        """
        Shortcut getter method for the number of items.
        """
        # rows = users, cols = items
        return self.current_state.shape[1]


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

    This class inherits from :class:`~base.base_components.BaseComponent`.

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

        attention_exp: float (optional, default: 0)
            If this parameter is non-zero, then the order of the items
            in the recommendation set affects the user's choice, in that
            the item chosen will be a function of its index in the recommendation
            set and the underlying user-item score. (See Chaney et al. 2018
            for a description of this mechanism.) Concretely, the item chosen will
            be according to
            :math:`i_u(t)=\\mathrm{argmax}_i( \\mathrm{rank}_{u,t}(i)^{\\alpha}
            \\cdot S_{u,i}(t) )`, where :math:`\\alpha` is the attention exponent
            and :math:`S_{u,i}(t)` is the underlying user-item score.

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

        repeat_interactions: bool (optional, default: True)
            If `True`, then users will interact with items regardless of whether
            they have already interacted with them before. If `False`, users
            will not perform repeat interactions.

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
        score_fn=mo.inner_product,
        verbose=False,
        seed=None,
        attention_exp=0.0,
        repeat_interactions=True,
    ):  # pylint: disable=too-many-arguments
        self.rng = Generator(seed=seed)
        # general input checks
        if actual_user_profiles is not None:
            if not isinstance(actual_user_profiles, (list, np.ndarray, sp.spmatrix)):
                raise TypeError("actual_user_profiles must be a list or numpy.ndarray")
        if interact_with_items is not None and not callable(interact_with_items):
            raise TypeError("interact_with_items must be callable")
        if actual_user_profiles is None and size is None:
            raise ValueError("actual_user_profiles and size can't both be None")
        if actual_user_profiles is None and not isinstance(size, tuple):
            raise TypeError("size must be a tuple, is %s" % type(size))
        if actual_user_scores is not None:
            if not isinstance(actual_user_scores, (list, np.ndarray, sp.spmatrix)):
                raise TypeError("actual_user_profiles must be a list or numpy.ndarray")
            actual_user_scores = ActualUserScores(actual_user_scores)
        if actual_user_profiles is None and size is not None:
            row_zeros = np.zeros(size[1])  # one row vector of zeroes
            while actual_user_profiles is None or mo.contains_row(actual_user_profiles, row_zeros):
                # generate matrix until no row is the zero vector
                actual_user_profiles = self.rng.normal(size=size)

        # check_consistency also returns num_items and num_attributes, which are not needed
        num_users = check_consistency(
            users=actual_user_profiles, user_item_scores=actual_user_scores, num_users=num_users
        )[0]
        self.actual_user_profiles = ActualUserProfiles(actual_user_profiles)
        self.interact_with_items = interact_with_items
        self.drift = drift
        self.attention_exp = attention_exp
        assert callable(score_fn)
        self.score_fn = score_fn  # function that dictates how scores will be generated
        self.actual_user_scores = actual_user_scores
        self.user_vector = np.arange(num_users, dtype=int)
        self.repeat_interactions = repeat_interactions
        if not repeat_interactions:
            self.user_interactions = np.array([], dtype=int).reshape((num_users, 0))
        self.name = "actual_user_scores"
        BaseComponent.__init__(self, verbose=verbose, init_value=self.actual_user_profiles.value)

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
            user_profiles=self.actual_user_profiles.value, item_attributes=item_attributes
        )
        if self.actual_user_scores is None:
            self.actual_user_scores = ActualUserScores(actual_scores)
        else:
            self.actual_user_scores.value = actual_scores

        self.actual_user_scores.store_state()

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
            user_profiles=self.actual_user_profiles.value, item_attributes=new_items
        )
        self.actual_user_scores.append_new_scores(new_scores)
        self.actual_user_scores.store_state()

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
        if not self.repeat_interactions:
            # "remove" items that have been interacted with by setting scores to negative infinity
            self.actual_user_scores.set_item_scores_to_value(self.user_interactions, float("-inf"))
        rec_item_scores = self.actual_user_scores.get_item_scores(items_shown)
        if self.attention_exp != 0:
            idxs = np.arange(items_shown.shape[1]) + 1
            multiplier = np.power(idxs, self.attention_exp)
            # multiply each row by the attention coefficient
            rec_item_scores = rec_item_scores * multiplier
        sorted_user_preferences = mo.argmax(rec_item_scores, axis=1)
        interactions = items_shown[self.user_vector, sorted_user_preferences]
        # logging information if requested
        if self.is_verbose():
            self.log(f"User scores for given items are:\n{str(rec_item_scores)}")
            self.log(f"Users interact with the following items respectively:\n{str(interactions)}")
        if self.drift > 0:
            if item_attributes is None:
                raise ValueError("Item attributes can't be None if user preferences are dynamic")
            # update user profiles based on the attributes of items they
            # interacted with
            interact_attrs = item_attributes.T[interactions, :]
            self.update_profiles(interact_attrs)
            # update user scores
            self.compute_user_scores(item_attributes)
        # record interactions if needed to ensure users don't repeat interactions
        if not self.repeat_interactions:
            interactions_col = interactions.reshape((-1, 1))
            # append interactions as column of user interactions
            self.user_interactions = np.hstack([self.user_interactions, interactions_col])
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
        self.actual_user_profiles.value = mo.slerp(
            self.actual_user_profiles, item_attributes, perc=self.drift
        )

    def store_state(self):
        """ Store the actual user scores in the state history """
        self.state_history.append(np.copy(self.actual_user_scores.value))


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
        score_fn=mo.inner_product,
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
        item_utilities = mo.to_dense(self.calc_dn_utilities(interaction_scores))
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
                The transformed utility values (i.e., :math:`z(\\textbf{v})`).
        """
        summed_norms = np.linalg.norm(user_item_scores, ord=self.beta, axis=1)
        denom = self.sigma + np.multiply(self.omega, summed_norms)
        return np.divide(user_item_scores.T, denom)  # now |I| x |U|

    def calc_dn_utilities(self, user_item_scores):
        """
        Scores items according to divisive normalization. Note that the parameters
        / matrix operations we perform here are directly taken from
        https://github.com/UofT-Neuroecon-1/Normalization. For more information,
        see Webb, R., Glimcher, P. W., & Louie, K. (2020). The Normalization of
        Consumer Valuations: Context-Dependent Preferences from Neurobiological
        Constraints. Management Science.

        Note that the generalized DN model takes the following functional form:
        :math:`z_i(\\textbf{v})=\\frac{v_i}{\\sigma+\\omega(\\sum_n v_n^{\\beta})^
        {\\frac{1}{\\beta}}}`, where :math:`\\sigma, \\omega, \\beta` are all
        parameters that specify the exact choice model. After the original values
        :math:`\\textbf{v}` are transformed this way, the choice is determined by
        choosing the maximum value over :math:`z(\\textbf{v})+\\mathbf{\\eta}`,
        which in our case is generated by a multivariate normal distribution.

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
        eps = self.sample_from_error_dist(num_choices, num_users)
        utility = normed_values + eps
        # transform so |U| x |I|
        return utility.T

    def sample_from_error_dist(self, num_choices, num_users):
        """
        The second stage of generating the divisive normalization utilities
        :math:`interactions_{u(t)} is adding the error term
        :math:`\\textbf{\\eta}`. In this implementation, we sample from
        a specific multivariate normal distribution used by Webb et al.
        (see https://github.com/UofT-Neuroecon-1/Normalization).

        Parameters
        -----------

        num_choices: int
            Number of items every user is choosing between.

        num_users: int
            Number of users in the system.

        Returns
        --------

        eps: :obj:`numpy.ndarray`
            Randomly sampled errors from the error distribution. Should have
            shape :math:`|I|\\times|U|`.
        """
        mean = np.zeros(num_choices)
        # in accordance with the DN model from Webb et al.,
        # the following covariance matrix has the structure
        # [ 1     0.5   ...   0.5   0.5 ]
        # [ 0.5    1    ...   0.5   0.5 ]
        # [ 0.5   0.5   ...    1    0.5 ]
        # [ 0.5   0.5   ...   0.5    1  ]
        cov = np.ones((num_choices, num_choices)) * 0.5
        cov[np.arange(num_choices), np.arange(num_choices)] = 1
        # generate |I| x |U| multivariate normal
        eps = self.rng.multivariate_normal(mean, cov, size=num_users).T
        return eps
