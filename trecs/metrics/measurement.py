"""
Set of various measurements that can be used to track outcomes of interest
throughout a simulation
"""
from abc import ABC, abstractmethod
import networkx as nx
from networkx import wiener_index
import numpy as np
from trecs.logging import VerboseMode
from trecs.components import (
    BaseObservable,
    register_observables,
)


class Measurement(BaseObservable, VerboseMode, ABC):
    """
    Abstract observable class to store measurements.

    Parameters
    -----------

        verbose: bool (optional, default: False)
            If True, enables verbose mode. Disabled by default.

        init_value: array_like or None or int or float (optional, default: None)
            The value of the metric before the start of the simulation.

    Attributes
    -----------

        measurement_history: list
            List of measurements. A new element is added at each timestep.

        name: str
            Name of the measurement quantity.
    """

    def __init__(self, name, verbose=False, init_value=None):
        self.name = name
        VerboseMode.__init__(self, __name__.upper(), verbose)
        self.measurement_history = list()
        if isinstance(init_value, np.ndarray):
            init_value = np.copy(init_value)
        self.measurement_history.append(init_value)

    def get_measurement(self):
        """
        Returns measurements. See
        :func:`~components.base_components.BaseObservable.get_observable`
        for more details.

        Returns
        --------
            Measurements: dict
        """
        return self.get_observable(data=self.measurement_history)

    def observe(self, observation, copy=True):  # pylint: disable=arguments-differ
        """
        Stores measurements. It can be called by implementations to ensure
        consistency when storing different measurements.

        Parameters
        -----------

        observation: array_like or int or float or None
            Element that will be stored

        copy: bool (optional, default: True)
            If True, the function stores a copy of observation. Useful for
            :obj:`numpy.ndarray`.

        """
        if copy:
            to_append = np.copy(observation)
        else:
            to_append = observation
        self.measurement_history.append(to_append)

    @abstractmethod
    def measure(self, recommender, **kwargs):
        """Function that should calculate some outcome of interest of the system
        at the current timestep
        """

    def get_timesteps(self):
        """
        Returns the number of measurements stored (which is equivalent to the
        number of timesteps that the system has been measuring).

        Returns
        --------

            Length of measurement_history: int
        """
        return len(self.measurement_history)


class MeasurementModule:  # pylint: disable=too-few-public-methods
    """
    Mixin for observers of :class:`Measurement` observables. Implements the
    `Observer design pattern`_.

    .. _`Observer design pattern`: https://en.wikipedia.org/wiki/Observer_pattern

    This mixin allows the system to monitor metrics. That is, at each timestep,
    an element will be added to the
    :attr:`~metrics.measurement.Measurement.measurement_history` lists of each
    metric that the system is monitoring.

    Attributes
    ------------

        metrics: list
            List of metrics that the system will monitor.

    """

    def __init__(self):
        self.metrics = list()

    def add_metrics(self, *args):
        """
        Adds metrics to the :attr:`metrics` list. This allows the system to
        monitor these metrics.

        Parameters
        -----------

            args: :class:`~metrics.measurement.Measurement`
                Accepts a variable number of metrics that inherits from
                :class:`~metrics.measurement.Measurement`
        """
        register_observables(
            observer=self.metrics, observables=list(args), observable_type=Measurement
        )


class InteractionMeasurement(Measurement):
    """
    Keeps track of the interactions between users and items.

    Specifically, at each timestep, it stores a histogram of length `|I|`, where
    element `i` is the number of interactions received by item `i`.

    Parameters
    -----------

        verbose: bool (optional, default: False)
            If True, enables verbose mode. Disabled by default.

    Attributes
    -----------
        Inherited by Measurement: :class:`.Measurement`

        name: str (optional, default: "interaction_histogram")
            Name of the measurement component.
    """

    def __init__(self, name="interaction_histogram", verbose=False):
        Measurement.__init__(self, name, verbose, init_value=None)

    @staticmethod
    def _generate_interaction_histogram(interactions, num_users, num_items):
        """
        Generates a histogram of the number of interactions per item at the
        given timestep.

        Parameters
        -----------

            interactions : :obj:`numpy.ndarray`
                Array of user interactions.

            num_users : int
                Number of users in the system

            num_items : int
                Number of items in the system

        Returns
        ---------
            Histogram : :obj:`numpy.ndarray`
                Histogram of the number of interactions aggregated by items at the given timestep.
        """
        histogram = np.zeros(num_items)
        np.add.at(histogram, interactions, 1)
        # Check that there's one interaction per user
        assert histogram.sum() == num_users
        return histogram

    def measure(self, recommender, **kwargs):
        """
        Measures and stores a histogram of the number of interactions per
        item at the given timestep.

        Parameters
        ------------
            recommender: :class:`~models.recommender.BaseRecommender`
                Model that inherits from :class:`~models.recommender.BaseRecommender`.

            **kwargs
                Keyword arguments, one of which must be `interactions`.
                `interactions` is a non-aggregated array of interactions --
                that is, an array of length `|U|` s.t. element `u` is the index
                of the item with which user `u` interacted.
        """
        interactions = kwargs.pop("interactions", None)
        histogram = self._generate_interaction_histogram(
            interactions, recommender.num_users, recommender.num_items
        )
        # histogram[::-1].sort()
        self.observe(histogram, copy=True)


class InteractionSimilarity(Measurement):
    """
    Keeps track of the average Jaccard similarity between interactions with items
    between pairs of users at each timestep. The pairs of users must be passed
    in by the user.

    Parameters
    -----------
        pairs: iterable of tuples
            Contains tuples representing each pair of users. Each user should
            be represented as an index into the user profiles matrix.

        verbose: bool (optional, default: False)
            If True, enables verbose mode. Disabled by default.

    Attributes
    -----------
        Inherited by Measurement: :class:`.Measurement`

        name: str (optional, default: "interaction_similarity")
            Name of the measurement component.
    """

    def __init__(self, pairs, name="interaction_similarity", verbose=False):
        self.pairs = pairs
        # will eventually be a matrix where each row corresponds to 1 user
        self.interaction_hist = None
        Measurement.__init__(self, name, verbose, init_value=None)

    def measure(self, recommender, **kwargs):
        """
        Measures the average Jaccard index of items that pairs of users have interacted
        with in the system. Intuitively, a higher average Jaccard index corresponds to
        increasing "homogenization" in that user behavior is becoming more and more
        similar (i.e., users have all interacted with the same items).

        Parameters
        ------------
            recommender: :class:`~models.recommender.BaseRecommender`
                Model that inherits from
                :class:`~models.recommender.BaseRecommender`.

            **kwargs
                Keyword arguments, one of which must be `interactions`, a |U| x
                1 array that contains the index of the items that each user has
                interacted with at this timestep.
        """
        similarity = 0
        interactions = kwargs.pop("interactions", None)
        if interactions is None:
            raise ValueError(
                "interactions must be passed in to InteractionSimilarity's `measure` "
                "method as a keyword argument"
            )

        if self.interaction_hist is None:
            self.interaction_hist = np.copy(interactions).reshape((-1, 1))
        else:
            self.interaction_hist = np.hstack(
                [self.interaction_hist, interactions.reshape((-1, 1))]
            )
        for pair in self.pairs:
            itemset_1 = set(self.interaction_hist[pair[0], :])
            itemset_2 = set(self.interaction_hist[pair[1], :])
            common = len(itemset_1.intersection(itemset_2))
            union = len(itemset_1.union(itemset_2))
            similarity += common / union / len(self.pairs)
        self.observe(similarity)


class RecSimilarity(Measurement):
    """
    Keeps track of the average Jaccard similarity between items seen by pairs
    of users at each timestep. The pairs of users must be passed in by the
    user.

    Parameters
    -----------
        pairs: iterable of tuples
            Contains tuples representing each pair of users. Each user should
            be represented as an index into the user profiles matrix.

        verbose: bool (optional, default: False)
            If True, enables verbose mode. Disabled by default.

    Attributes
    -----------
        Inherited by Measurement: :class:`.Measurement`

        name: str (optional, default: "rec_similarity")
            Name of the measurement component.
    """

    def __init__(self, pairs, name="rec_similarity", verbose=False):
        self.pairs = pairs
        Measurement.__init__(self, name, verbose, init_value=None)

    def measure(self, recommender, **kwargs):
        """
        Measures the average Jaccard index of items shown to pairs of users in
        the system. Intuitively, a higher average Jaccard index corresponds to
        increasing "homogenization" in that the recommender system is starting
        to treat each user the same way (i.e., show them the same items).

        Parameters
        ------------
            recommender: :class:`~models.recommender.BaseRecommender`
                Model that inherits from
                :class:`~models.recommender.BaseRecommender`.

            **kwargs
                Keyword arguments, one of which must be `items_shown`, a |U| x
                num_items_per_iter matrix that contains the indices of every
                item shown to every user at a particular timestep.
        """
        similarity = 0
        items_shown = kwargs.pop("items_shown", None)
        if items_shown is None:
            raise ValueError(
                "items_shown must be passed in to RecSimilarity's `measure` "
                "method as a keyword argument"
            )
        for pair in self.pairs:
            itemset_1 = set(items_shown[pair[0], :])
            itemset_2 = set(items_shown[pair[1], :])
            common = len(itemset_1.intersection(itemset_2))
            union = len(itemset_1.union(itemset_2))
            similarity += common / union / len(self.pairs)
        self.observe(similarity)


class HomogeneityMeasurement(InteractionMeasurement):
    """
    Measures the homogeneity of the interactions between users and items.

    Specifically, at each timestep, it measures whether interactions are spread
    among many items or only a few items.

    This class inherits from :class:`.InteractionMeasurement`.

    Parameters
    -----------

        verbose: bool (optional, default: False)
            If True, enables verbose mode. Disabled by default.

    Attributes
    -----------
        Inherited by InteractionMeasurement: :class:`.InteractionMeasurement`

        name: str (optional, default: "homogeneity")
            Name of the measurement component.

        _old_histogram: None, list, array_like
            A copy of the histogram at the previous timestep.
    """

    def __init__(self, verbose=False):
        self.histogram = None
        self._old_histogram = None
        InteractionMeasurement.__init__(self, name="homogeneity", verbose=verbose)

    def measure(self, recommender, **kwargs):
        """
        Measures the homogeneity of user interactions -- that is, whether
        interactions are spread among many items or only a few items.

        Parameters
        ------------
            recommender: :class:`~models.recommender.BaseRecommender`
                Model that inherits from
                :class:`~models.recommender.BaseRecommender`.

            **kwargs
                Keyword arguments, one of which must be `interactions`.
                `interactions` is a non-aggregated array of interactions --
                that is, an array of length `|U|` s.t. element `u` is the index
                of the item with which user `u` interacted.
        """
        interactions = kwargs.pop("interactions", None)
        assert interactions.size == recommender.num_users
        histogram = self._generate_interaction_histogram(
            interactions, recommender.num_users, recommender.num_items
        )
        histogram[::-1].sort()
        if self._old_histogram is None:
            self._old_histogram = np.zeros(recommender.num_items)
        self.observe(np.trapz(self._old_histogram, dx=1) - np.trapz(histogram, dx=1), copy=False)
        self._old_histogram = np.copy(histogram)
        self.histogram = histogram


class MSEMeasurement(Measurement):
    """
    Measures the mean squared error (MSE) between real and predicted user scores.

    It can be used to evaluate how accurate the model predictions are.

    This class inherits from :class:`.Measurement`.

    Parameters
    -----------

        verbose: bool (optional, default: False)
            If True, enables verbose mode. Disabled by default.

    Attributes
    -----------
        Inherited by Measurement: :class:`.Measurement`

        name: str (optional, default: "mse")
            Name of the measurement component.
    """

    def __init__(self, verbose=False):
        Measurement.__init__(self, "mse", verbose=verbose, init_value=None)

    def measure(self, recommender, **kwargs):
        """
        Measures and records the mean squared error between the user preferences
        predicted by the system and the users' actual preferences.

        Parameters
        ------------
            recommender: :class:`~models.recommender.BaseRecommender`
                Model that inherits from :class:`~models.recommender.BaseRecommender`.

            **kwargs
                Keyword arguments, one of which must be `interactions`.
                `interactions` is a non-aggregated array of interactions --
                that is, an array of length `|U|` s.t. element `u` is the index
                of the item with which user `u` interacted.
        """
        diff = recommender.predicted_scores - recommender.users.actual_user_scores
        self.observe((diff ** 2).mean(), copy=False)


class DiffusionTreeMeasurement(Measurement):
    """
    Class that implements an information diffusion tree. The current
    implementation assumes that agents using this class (i.e., a model)
    implement an :attr:`~models.bass.BassModel.infection_state` matrix that
    denotes the initial state of information.

    In this implementation, the nodes represent users and are labeled with the
    user indices. A branch between nodes `u` and `v` indicates that user `u`
    passed information onto user `v` -- that is, `u` "infected" `v`.

    Trees are implemented using the `Networkx library`_. Please refer to
    Networkx's `documentation`_ for more details.

    .. _Networkx library: http://networkx.github.io
    .. _documentation: https://networkx.github.io/documentation/stable/

    Parameters
    -----------

        infection_state: array_like
            The initial "infection state"

        verbose: bool (optional, default: False)
            If True, enables verbose mode. Disabled by default.

    Attributes
    -----------
        Inherited by Measurement: :class:`.Measurement`

        name: str (optional, default: "num_infected")
            Name of the metric that is recorded at each time step. Note that,
            in this case, the metric stored in
            :attr:`~.Measurement.measurement_history` is actually the
            **number of infected users**. The diffusion tree itself is kept in
            the :attr:`.diffusion_tree` data structure.

        diffusion_tree: :obj:`networkx.Graph`
            Diffusion tree.

        _old_infection_state: array_like
            Infection state at the previous timestep.
    """

    def __init__(self, infection_state, verbose=False):
        self._old_infection_state = None
        self.diffusion_tree = nx.Graph()
        self._manage_new_infections(None, np.copy(infection_state))
        self._old_infection_state = np.copy(infection_state)
        Measurement.__init__(
            self, "num_infected", verbose=verbose, init_value=self.diffusion_tree.number_of_nodes()
        )

    def _find_parents(self, user_profiles, new_infected_users):
        """ Find the users who infected the newly infected users """
        if (self._old_infection_state == 0).all():
            # Node is root
            return None
        # TODO: function is_following() based on code below:
        # candidates must have been previously infected
        prev_infected_users = np.where(self._old_infection_state > 0)[0]
        # candidates must be connected to newly infected users
        candidate_parents = user_profiles[:, prev_infected_users][new_infected_users]
        parents = prev_infected_users[np.argsort(candidate_parents)[:, -1]]
        return parents

    def _add_to_graph(self, user_profiles, new_infected_users):
        """Add the newly infected users to the graph with edges to the users
        who infected them
        """
        self.diffusion_tree.add_nodes_from(new_infected_users)
        parents = self._find_parents(user_profiles, new_infected_users)
        # connect parent(s) and child(ren)
        if parents is not None:
            edges = np.vstack((parents, new_infected_users)).T
            self.diffusion_tree.add_edges_from(edges)

    def _manage_new_infections(self, user_profiles, current_infection_state):
        """Add new infected users to graph and return number of newly infected
        users
        """
        if self._old_infection_state is None:
            self._old_infection_state = np.zeros(current_infection_state.shape)
        new_infections = current_infection_state - self._old_infection_state
        if (new_infections == 0).all():
            # no new infections
            return 0
        new_infected_users = np.where(new_infections > 0)[0]  # only the rows
        self._add_to_graph(user_profiles, new_infected_users)
        # return number of new infections
        return new_infected_users.shape[0]

    def measure(self, recommender, **kwargs):
        """
        Updates tree with new infections and stores information about new
        infections. In :attr:`~.Measurement.measurement_history`, it stores the
        total number of infected users in the system -- that is, the number of
        nodes in the tree.

        Parameters
        ------------
            recommender: :class:`~models.recommender.BaseRecommender`
                Model that inherits from :class:`~models.recommender.BaseRecommender`.

            **kwargs
                Keyword arguments, one of which must be `interactions`.
                `interactions` is a non-aggregated array of interactions --
                that is, an array of length `|U|` s.t. element `u` is the index
                of the item with which user `u` interacted.
        """
        self._manage_new_infections(recommender.users_hat, recommender.infection_state)
        self.observe(self.diffusion_tree.number_of_nodes(), copy=False)
        self._old_infection_state = np.copy(recommender.infection_state)

    def draw_tree(self):
        """
        Plots the tree using the Networkx library API.
        """
        nx.draw(self.diffusion_tree, with_labels=True)


class StructuralVirality(DiffusionTreeMeasurement):
    """
    This class extends :class:`DiffusionTreeMeasurement` with the concept of
    structural virality developed by Goel, Anderson, Hofman, and Watts in
    `The Structural Virality of Online Diffusion`_. It is used in
    :class:`~models.bass.BassModel`.

    .. _The Structural Virality of Online Diffusion: https://5harad.com/papers/twiral.pdf

    Parameters
    ----------

        infection_state: array_like
            The initial "infection state" (see :class:`DiffusionTreeMeasurement`).

    """

    def __init__(self, infection_state, verbose=False):
        DiffusionTreeMeasurement.__init__(self, infection_state, verbose)

    def get_structural_virality(self):
        """
        Returns a measure of structural virality.

        Returns
        --------
            Structural virality: float
        """
        num_nodes = self.diffusion_tree.number_of_nodes()
        return wiener_index(self.diffusion_tree) / (num_nodes * (num_nodes - 1))


class AverageFeatureScoreRange(Measurement):
    """
    Measures the average range (across users) of item attributes for items
    users chose to interact with at a time step.

    This metric is based on the item diversity measure used in :

        Willemsen, M. C., Graus, M. P.,
        & Knijnenburg, B. P. (2016). Understanding the role of latent feature
        diversification on choice difficulty and satisfaction. User Modeling
        and User-Adapted Interaction, 26(4), 347-389.

    This class inherits from :class:`.InteractionMeasurement`.

    Parameters
    -----------

        verbose: bool (optional, default: False)
            If True, enables verbose mode. Disabled by default.

    Attributes
    -----------
        Inherited by Measurement: :class:`.Measurement`

        name: str (optional, default: "afsr")
            Name of the measurement component.
    """

    def __init__(self, name="afsr", verbose=False):
        Measurement.__init__(self, name, verbose, init_value=None)

    def measure(self, recommender, **kwargs):
        """
        Measures the average range (across users) of item attributes for items
        users chose to interact with at a time step.

        Parameters
        ------------
            recommender: :class:`~models.recommender.BaseRecommender`
                Model that inherits from
                :class:`~models.recommender.BaseRecommender`.

            **kwargs
                Keyword arguments, one of which must be `interactions`.
                `interactions` is a non-aggregated array of interactions --
                that is, an array of length `|U|` s.t. element `u` is the index.ide
                of the item with which user `u` interacted.
        """
        interactions = kwargs.pop("interactions", None)
        assert interactions.size == recommender.num_users
        interacted_item_attr = recommender.items_hat[:, interactions]

        if {item for sublist in interacted_item_attr for item in sublist} == {0, 1}:
            raise ValueError("AFSR is not intended for binary features.")

        afsr = (
            sum(interacted_item_attr.max(axis=0) - interacted_item_attr.min(axis=0))
            / interacted_item_attr.shape[0]
        )

        self.observe(afsr)
