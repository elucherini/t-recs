from abc import ABC, abstractmethod
from rec.utils import VerboseMode
from rec.components import BaseObservable
import numpy as np


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
    """

    def __init__(self, verbose=False, init_value=None):
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

    def observe(self, observation, copy=True):
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
    def measure(self, step, interactions, recommender):
        pass

    def get_timesteps(self):
        """
        Returns the number of measurements stored (which is equivalent to the
        number of timesteps that the system has been measuring).

        Returns
        --------

            Length of measurement_history: int
        """
        return len(self.measurement_history)


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

        name: str
            Name of the measurement component.
    """

    def __init__(self, verbose=False):
        self.name = "interaction_histogram"
        Measurement.__init__(self, verbose, init_value=None)

    def _generate_interaction_histogram(self, interactions, num_users, num_items):
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

    def measure(self, step, interactions, recommender):
        """
            Measures and stores a histogram of the number of interactions per
            item at the given timestep.

            Parameters
            ------------

                step: int
                    Current time step

                interactions: :obj:`numpy.array`
                    Non-aggregated array of interactions -- that is, an array of
                    length `|U|` s.t. element `u` is the index of the item with
                    which user `u` interacted.

                recommender: :class:`~models.recommender.BaseRecommender`
                    Model that inherits from :class:`~models.recommender.BaseRecommender`.
        """

        histogram = self._generate_interaction_histogram(
            interactions, recommender.num_users, recommender.num_items
        )
        # histogram[::-1].sort()
        self.observe(histogram, copy=True)


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

        name: str
            Name of the measurement component.

        _old_histogram: None, list, array_like
            A copy of the histogram at the previous timestep.
    """

    def __init__(self, verbose=False):
        self.histogram = None
        self._old_histogram = None
        self.name = "homogeneity"
        Measurement.__init__(self, verbose, init_value=None)

    def measure(self, step, interactions, recommender):
        """
            Measures the homogeneity of user interactions -- that is, whether
            interactions are spread among many items or only a few items.

            Parameters
            ------------

                step: int
                    Current time step

                interactions: :obj:`numpy.array`
                    Non-aggregated array of interactions -- that is, an array of
                    length `|U|` s.t. element `u` is the index of the item with
                    which user `u` interacted.

                recommender: :class:`~models.recommender.BaseRecommender`
                    Model that inherits from
                    :class:`~models.recommender.BaseRecommender`.
        """
        assert interactions.size == recommender.num_users
        histogram = self._generate_interaction_histogram(
            interactions, recommender.num_users, recommender.num_items
        )
        histogram[::-1].sort()
        if self._old_histogram is None:
            self._old_histogram = np.zeros(recommender.num_items)
        self.observe(
            np.trapz(self._old_histogram, dx=1) - np.trapz(histogram, dx=1), copy=False
        )
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

        name: str
            Name of the measurement component.
    """

    def __init__(self, verbose=False):
        self.name = "mse"
        Measurement.__init__(self, verbose, init_value=None)

    def measure(self, step, interactions, recommender):
        """
        Measures and records the mean squared error between the user preferences
        predicted by the system and the users' actual preferences.

        Parameters
        ------------

            step: int
                Current time step

            interactions: :obj:`numpy.array`
                Non-aggregated array of interactions -- that is, an array of
                length `|U|` s.t. element `u` is the index of the item with
                which user `u` interacted.

            recommender: :class:`~models.recommender.BaseRecommender`
                Model that inherits from :class:`~models.recommender.BaseRecommender`.
        """
        diff = (
            recommender.predicted_scores - recommender.users.actual_user_scores
        )
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

        name: str
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
        import networkx as nx

        self.name = "num_infected"
        self._old_infection_state = None
        self.diffusion_tree = nx.Graph()
        self._manage_new_infections(None, np.copy(infection_state))
        self._old_infection_state = np.copy(infection_state)
        Measurement.__init__(
            self, verbose, init_value=self.diffusion_tree.number_of_nodes()
        )

    def _find_parents(self, user_profiles, new_infected_users):
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
        self.diffusion_tree.add_nodes_from(new_infected_users)
        parents = self._find_parents(user_profiles, new_infected_users)
        # connect parent(s) and child(ren)
        if parents is not None:
            edges = np.vstack((parents, new_infected_users)).T
            self.diffusion_tree.add_edges_from(edges)

    def _manage_new_infections(self, user_profiles, current_infection_state):
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

    def measure(self, step, interactions, recommender):
        """
        Updates tree with new infections and stores information about new
        infections. In :attr:`~.Measurement.measurement_history`, it stores the
        total number of infected users in the system -- that is, the number of
        nodes in the tree.

        Parameters
        ------------

            step: int
                Current time step

            interactions: :obj:`numpy.array`
                Non-aggregated array of interactions -- that is, an array of
                length `|U|` s.t. element `u` is the index of the item with
                which user `u` interacted.

            recommender: :class:`~models.recommender.BaseRecommender`
                Model that inherits from :class:`~models.recommender.BaseRecommender`.
        """
        num_new_infections = self._manage_new_infections(
            recommender.users_hat, recommender.infection_state
        )
        self.observe(self.diffusion_tree.number_of_nodes(), copy=False)
        self._old_infection_state = np.copy(recommender.infection_state)

    def draw_tree(self):
        """
        Plots the tree using the Networkx library API.
        """
        import matplotlib.pyplot as plt
        import networkx as nx

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
        import networkx as nx
        from networkx.algorithms.wiener import wiener_index

        n = self.diffusion_tree.number_of_nodes()
        return nx.wiener_index(self.diffusion_tree) / (n * (n - 1))
