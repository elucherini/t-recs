import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm
from rec import utils
from rec.metrics import MSEMeasurement, Measurement
from rec.components import Users, Items, PredictedScores, PredictedUserProfiles
from rec.components import BaseObserver, BaseComponent, FromNdArray
from rec.utils import VerboseMode
from rec.random import Generator


class MeasurementModule(BaseObserver):
    """
    Mixin for observers of :class:`Measurement` observables.Implements the
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
        self.register_observables(
            observer=self.metrics, observables=list(args), observable_type=Measurement
        )


class SystemStateModule(BaseObserver):
    """
    Mixin for observers of :class:`Component` observables. Implements the
    `Observer design pattern`_.

    .. _`Observer design pattern`: https://en.wikipedia.org/wiki/Observer_pattern

    This mixin allows the system to monitor the system state. That is, at each
    timestep, an element will be added to the
    :attr:`~components.base_components.BaseComponent.state_history` lists 
    of each component that the system is monitoring.

    Attributes
    ------------

        _system_state: list
            List of system state components that the system will monitor.

    """

    def __init__(self, components=None):
        self._system_state = list()

    def add_state_variable(self, *args):
        """
        Adds metrics to the :attr:`_system_state` list. This allows the system
        to monitor these system state components.

        Parameters
        -----------

            args: :class:`~components.base_components.BaseComponent`
                Accepts a variable number of components that inherit from class
                :class:`~components.base_components.BaseComponent`
        """
        self.register_observables(
            observer=self._system_state,
            observables=list(args),
            observable_type=BaseComponent,
        )


class BaseRecommender(MeasurementModule, SystemStateModule, VerboseMode, ABC):
    """Abstract class representing a recommender system.

        The attributes and methods in this class can be generalized beyond
        recommender systems and are currently common to all pre-loaded models.

        Parameters
        -----------

            users_hat: :obj:`numpy.ndarray`
                An array representing users. The shape and meaning depends on
                the implementation of the concrete class.

            items_hat: :obj:`numpy.ndarray`
                An array representing items. The shape and meaning depends on
                the implementation of the concrete class.

            users: :obj:`numpy.ndarray` or :class:`~components.users.Users`
                An array representing real user preferences unknown to the
                system. Shape is |U| x |A|, where |A| is the number of attributes
                and |U| is the number of users.

            items: :obj:`numpy.ndarray` or :class:`~components.items.Items`
                An array representing real item attributes unknown to the
                system. Shape is |A| x |I|, where |I| is the number of items
                and |A| is the number of attributes.

            num_users: int
                The number of users in the system.

            num_items: int
                The number of items in the system.

            num_items_per_iter: int
                Number of items presented to the user at each iteration.

            measurements: list
                List of metrics to monitor.

            system_state: list
                List of system state components to monitor.

            verbose: bool (optional, default: False)
                If True, it enables verbose mode.

            seed: int, None (optional, default: None)
                Seed for random generator used

        Attributes
        -----------

            users_hat: :class:`~components.users.PredictedUserProfiles`
                An array representing users, matching user_representation. The
                shape and meaning depends on the implementation of the concrete
                class.

            items_hat: :class:`~components.items.Items`
                An array representing items, matching item_representation. The
                shape and meaning depends on the implementation of the concrete
                class.

            users: :class:`~components.users.Users`
                An array representing real user preferences, matching
                actual_users. The shape and meaning depends on the
                implementation of the concrete class.

            items: :class:`~components.users.Users`
                An array representing real user preferences, matching
                actual_users. The shape and meaning depends on the
                implementation of the concrete class.

            predicted_scores: :class:`~components.users.PredictedScores`
                An array representing the user preferences as perceived by the
                system. The shape is always `|U|x|I|`, where `|U|` is the number
                of users in the system and `|I|` is the number of items in the
                system. The scores are calculated with the dot product of
                :attr:`users_hat` and :attr:`items_hat`.

            num_users: int
                The number of users in the system.

            num_items: int
                The number of items in the system.

            num_items_per_iter: int
                Number of items presented to the user per iteration.

            random_state: :class:`rec.random.generators.Generator`

            indices: :obj:`numpy.ndarray`
                A `|U|x|I|` array representing the past interactions of each
                user. This keeps track of which items each user has interacted
                with, so that it won't be presented to the user again if
                `repeated_items` are not allowed.
    """

    @abstractmethod
    def __init__(
        self,
        users_hat,
        items_hat,
        users,
        items,
        num_users,
        num_items,
        num_items_per_iter,
        measurements=None,
        system_state=None,
        verbose=False,
        seed=None,
    ):
        # Init logger
        VerboseMode.__init__(self, __name__.upper(), verbose)
        # Initialize measurements
        MeasurementModule.__init__(self)
        if measurements is not None:
            self.add_metrics(*measurements)
        # init the recommender system's internal representation of users
        # and items
        self.users_hat = PredictedUserProfiles(users_hat)
        self.items_hat = Items(items_hat)
        # set predicted scores
        self.predicted_scores = PredictedScores(
            self.train(self.users_hat, self.items_hat, normalize=True)
        )
        assert self.predicted_scores is not None

        if not utils.is_valid_or_none(num_users, int):
            raise TypeError("num_users must be an int")
        if not utils.is_valid_or_none(num_items, int):
            raise TypeError("num_items must be an int")
        if not utils.is_valid_or_none(num_items_per_iter, int):
            raise TypeError("num_items_per_iter must be an int")
        if not hasattr(self, "metrics"):
            raise ValueError("You must define at least one measurement module")

        # check users array
        if not utils.is_valid_or_none(
            users, (list, np.ndarray, Users)
        ):
            raise TypeError("users must be array_like or Users")
        if users is None:
            self.users = Users(
                size=self.users_hat.shape, num_users=num_users, seed=seed
            )
        if isinstance(users, (list, np.ndarray)):
            self.users = Users(
                actual_user_scores=users, num_users=num_users
            )
        if isinstance(users, Users):
            self.users = users

        # check items array
        if not utils.is_valid_or_none(
            items, (list, np.ndarray, Items)
        ):
            raise TypeError("items must be array_like or Items")
        if items is None:
            self.actual_items = Items(
                size=self.items_hat.shape, seed=seed
            )
        if isinstance(items, (list, np.ndarray)):
            # TODO: this doesn't work!!!!
            self.actual_items = Items(items)
        if isinstance(items, Items):
            self.actual_items = items

        # system state
        SystemStateModule.__init__(self)
        self.add_state_variable(
            self.users_hat,
            self.users,
            self.items_hat,
            self.predicted_scores,
        )
        if system_state is not None:
            self.add_state_variable(*system_state)

        self.users.compute_user_scores(self.train)

        assert self.users and isinstance(self.users, Users)
        self.num_users = num_users
        self.num_items = num_items
        self.num_items_per_iter = num_items_per_iter
        self.random_state = Generator(seed)
        # Matrix keeping track of the items consumed by each user
        self.indices = np.tile(np.arange(num_items), (num_users, 1))
        self.log("Recommender system ready")
        self.log("Num items: %d" % self.num_items)
        self.log("Users: %d" % self.num_users)
        self.log("Items per iter: %d" % self.num_items_per_iter)
        if seed is not None:
            self.log("Set seed to %d" % seed)
        else:
            self.log("Seed was not set.")

    def train(self, user_profiles=None, item_attributes=None, normalize=True):
        """
        Updates scores predicted by the system based on past interactions for
        better user predictions. Specifically, it updates :attr:`predicted_scores`
        with a dot product.

        Parameters
        -----------

            user_profiles: :obj:`array_like` or None (optional, default: None)
                First factor of the dot product, which should provide a
                representation of users. If None, the first factor defaults to
                :attr:`user_profiles`.

            item_attributes: :obj:`array_like` or None (optional, default: None)
                Second factor of the dot product, which should provide a
                representation of items. If None, the second factor defaults to
                :attr:`item_attributes`.

            normalize: bool (optional, default: True)
                Set to True if the scores should be normalized, False otherwise.

        Returns
        --------
            predicted_scores: :class:`~components.users.PredictedScores`
        """
        if user_profiles is None:
            user_profiles = self.users_hat
        if item_attributes is None:
            item_attributes = self.items_hat
        if normalize:
            user_profiles = utils.normalize_matrix(user_profiles, axis=1)
        assert user_profiles.shape[1] == item_attributes.shape[0]
        predicted_scores = np.dot(user_profiles, item_attributes)
        self.log(
            "System updates predicted scores given by users (rows) "
            + "to items (columns):\n"
            + str(predicted_scores)
        )
        assert predicted_scores is not None
        return predicted_scores

    def generate_recommendations(self, k=1, indices_prime=None):
        """
        Generate recommendations

        Parameters
        -----------

            k : int (optional, default: 1)
                Number of items to recommend.

            indices_prime : :obj:`numpy.ndarray` or None (optional, default: None)
                A matrix containing the indices of the items each user has
                interacted with. It is used to ensure that the user is presented
                with items they have already interacted with.

        Returns
        ---------
            Recommendations: :obj:`numpy.ndarray`
        """
        if indices_prime is None:
            indices_prime = self.indices[np.where(self.indices >= 0)]
            indices_prime = indices_prime.reshape((self.num_users, -1))
        if indices_prime.size == 0 or k > indices_prime.shape[1]:
            self.log("Insufficient number of items left!")
            indices_prime = self.indices[np.where(self.indices >= 0)]
            indices_prime = indices_prime.reshape((self.num_users, -1))
        row = np.repeat(self.users._user_vector, indices_prime.shape[1])
        row = row.reshape((self.num_users, -1))
        self.log("Row:\n" + str(row))
        self.log("Indices_prime:\n" + str(indices_prime))
        s_filtered = self.predicted_scores[row, indices_prime]
        permutation = s_filtered.argsort()
        rec = indices_prime[row, permutation]
        probabilities = np.logspace(0.0, rec.shape[1] / 10.0, num=rec.shape[1], base=2)
        probabilities = probabilities / probabilities.sum()
        self.log("Items ordered by preference for each user:\n" + str(rec))
        picks = self.random_state.choice(
            permutation.shape[1], p=probabilities, size=(self.num_users, k)
        )
        return rec[
            np.repeat(self.users._user_vector, k).reshape((self.num_users, -1)),
            picks,
        ]

    def recommend(self, startup=False):
        """
        Implements the recommendation process by combining recommendations and
        new (random) items.

        Parameters
        -----------
            startup (bool, optional): If True, the system is in "startup"
                (exploration) mode and only presents the user with new randomly
                chosen items. This is done to maximize exploration.

        Returns
        --------
            Items: :obj:`numpy.ndarray`
                New and recommended items in random order.
        """
        if startup:
            num_new_items = self.num_items_per_iter
            num_recommended = 0
        else:
            num_new_items = self.random_state.integers(0, self.num_items_per_iter)
            num_recommended = self.num_items_per_iter - num_new_items

        if num_recommended == 0 and num_new_items == 0:
            raise ValueError(
                "Not allowed for there to be 0 new items presented and 0"
                + " recommended items."
            )

        if num_recommended > 0:
            recommended = self.generate_recommendations(k=num_recommended)
            assert num_recommended == recommended.shape[1]
            assert recommended.shape[0] == self.num_users
            self.log(
                "System recommended these items (cols) to each user "
                + "(rows):\n"
                + str(recommended)
            )
        else:
            recommended = None
        indices_prime = self.indices[np.where(self.indices >= 0)]
        indices_prime = indices_prime.reshape((self.num_users, -1))
        # Current assumptions:
        # 1. Interleave new items and recommended items
        # 2. Each user interacts with one element depending on preference
        # 3. Users can't interact with the same item more than once
        assert np.count_nonzero(self.indices == -1) % self.num_users == 0
        self.log("Choice among %d items" % (indices_prime.shape[0]))
        if indices_prime.shape[1] < num_new_items:
            self.log("Insufficient number of items left!")
            indices_prime = self.indices[np.where(self.indices >= 0)]
            indices_prime = indices_prime.reshape((self.num_users, -1))

        if num_new_items:
            col = self.random_state.integers(
                indices_prime.shape[1], size=(self.num_users, num_new_items)
            )
            row = np.repeat(self.users._user_vector, num_new_items).reshape(
                (self.num_users, -1)
            )
            new_items = indices_prime[row, col]
            self.log(
                "System picked these items (cols) randomly for each user "
                + "(rows):\n"
                + str(new_items)
            )

        if num_recommended and num_new_items:
            items = np.concatenate((recommended, new_items), axis=1)
        elif num_new_items:
            items = new_items
        else:
            items = recommended
        self.random_state.shuffle(items.T)
        return items

    @abstractmethod
    def _update_user_profiles(self):
        """
        Updates user profiles based on last interaction.

        It must be defined in the concrete class.
        """
        pass

    def run(
        self, timesteps=50, startup=False, train_between_steps=True, repeated_items=True
    ):
        """
        Runs simulation for the given timesteps.

        Parameters
        -----------

            timestep : int (optional, default: 50)
                Number of timesteps for simulation.

            startup : bool (optional, default: False)
                If True, it runs the simulation in startup mode (see
                :func:`recommend` and :func:`startup_and_train`)

            train_between_steps : bool (optional, default: True)
                If True, the model is retrained after each timestep with the
                information gathered in the previous step.

            repeated_items : bool (optional, default: True)
                If True, repeated items are allowed in the system -- that is,
                users can interact with the same item more than once.
        """
        if not startup:
            self.log(
                "Run -- interleave recommendations and random items " + "from now on"
            )
        for t in tqdm(range(timesteps)):
            self.log("Step %d" % t)
            items = self.recommend(startup=startup)
            interactions = self.users.get_user_feedback(items=items)
            if not repeated_items:
                self.indices[self.users._user_vector, interactions] = -1
            self._update_user_profiles(interactions)
            self.log(
                "System updates user profiles based on last interaction:\n"
                + str(self.users_hat)
            )
            # train between steps:
            if train_between_steps:
                self.predicted_scores[:, :] = self.train(
                    self.users_hat, self.items_hat
                )
            self.measure_content(interactions, step=t)
        # If no training in between steps, train at the end:
        if not train_between_steps:
            self.predicted_scores[:, :] = self.train(
                self.users_hat, self.items_hat
            )
            self.measure_content(interactions, step=t)

    def startup_and_train(self, timesteps=50):
        """
        Runs simulation in startup mode by calling :func:`run` with
        startup=True. For more information about startup mode, see :func:`run`
        and :func:`recommend`.

        Parameters
        -----------

            timestep : int (optional, default: 50)
                Number of timesteps for simulation
        """
        self.log("Startup -- recommend random items")
        return self.run(timesteps, startup=True, train_between_steps=False)

    def _expand_items(self, num_new_items=None):
        """
        Increases number of items in the system.

        Parameters
        -----------
            num_new_items (int, optional): number of new items to add to the
                system. If None, it is equal to twice the number of items
                presented to the user in one iteration.
        """
        """
        if num_new_items is None:
            num_new_items = 2 * self.num_items_per_iter
        if not isinstance(num_new_items, int):
            raise TypeError("num_new_items should be int, is instead %s" % (
                                                                    type(num_new_items)))
        if num_new_items < 1:
            raise ValueError("Can't increment items by a number smaller than 1!")
        #new_indices = np.tile(self.items_hat.expand_items(self, num_new_items),
        #    (self.num_users,1))
        self.indices = np.concatenate((self.indices, new_indices), axis=1)
        self.users.compute_user_scores(self.train)
        self.predicted_scores = self.train(self.users_hat, self.items_hat)
        """
        pass

    def get_measurements(self):
        """
        Returns all available measurements. For more details, please see the 
        :class:`~metrics.measurement.Measurement` class.

        Returns
        --------
        Monitored measurements: dict
        """
        if len(self.metrics) < 1:
            raise ValueError("No measurement module defined")
        measurements = dict()
        for metric in self.metrics:
            measurements = {**measurements, **metric.get_measurement()}
        if "timesteps" not in measurements:
            # pick first measurement's length for # of timesteps since they're
            # going to be the same
            elapsed = np.arange(self.metrics[0].get_timesteps())
            measurements["timesteps"] = elapsed
        return measurements

    def get_system_state(self):
        """
        Return history of system state components stored in the
        :attr:`~components.base_components.BaseComponent.state_history` of the
        components stored in :attr:`.SystemStateModule._system_state`.

        Returns
        --------
            System state: dict
        """
        if len(self._system_state) < 1:
            raise ValueError("No measurement module defined")
        state = dict()
        for component in self._system_state:
            state = {**state, **component.get_component_state()}
        if "timesteps" not in state:
            # pick first measurement's length for # of timesteps since they're
            # going to be the same
            elapsed = np.arange(self._system_state[0].get_timesteps())
            state["timesteps"] = elapsed
        # FIXME: this is needed because Users.actual_user_scores is initialized to None
        if (
            "actual_user_scores" in state
            and "timesteps" in state
            and len(state["actual_user_scores"]) > len(state["timesteps"])
        ):
            state["actual_user_scores"].pop(0)
        return state

    def measure_content(self, interactions, step):
        """
        Calls method in the :class:`Measurements` module to record metrics.
        For more details, see the :class:`Measurements` class and its measure
        method.

        Parameters
        -----------
            interactions (:obj:`numpy.ndarray`): matrix of interactions
                per users at a given time step.

            step (int): step on which the recorded interactions refers to.
        """
        for metric in self.metrics:
            metric.measure(step, interactions, self)
        for component in self._system_state:
            component.store_state()
