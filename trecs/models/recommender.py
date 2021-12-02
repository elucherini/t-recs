"""
BaseRecommender, the foundational class for all recommender systems
implementable in our simulation library
"""
from abc import ABC, abstractmethod
import warnings
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from trecs.metrics import MeasurementModule
from trecs.base import SystemStateModule
from trecs.components import (
    Users,
    Items,
    Creators,
    PredictedScores,
    PredictedUserProfiles,
    PredictedItems,
)
from trecs.logging import VerboseMode
import trecs.matrix_ops as mo
from trecs.random import Generator
from trecs.utils import is_valid_or_none


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
            system. Shape is :math:`|U| \\times |A|`, where :math:`|A|` is the
            number of attributes and :math:`|U|` is the number of users. When
            a `numpy.ndarray` is passed in, we assume this represents the user
            *scores*, not the users' actual attribute vectors.

        items: :obj:`numpy.ndarray` or :class:`~components.items.Items`
            An array representing real item attributes unknown to the
            system. Shape is :math:`|A|\\times|I|`, where :math:`|I|` is the
            number of items and :math:`|A|` is the number of attributes.

        num_users: int
            The number of users in the system.

        num_items: int
            The number of items in the system.

        num_items_per_iter: int
            Number of items presented to the user at each iteration.

        measurements: list
            List of metrics to monitor.

        record_base_state: bool (optional, default: False)
            If True, the system will record at each time step its internal
            representation of users profiles and item profiles, as well as the
            true user profiles and item profiles. It will also record the
            predicted user-item scores at each time step.

        system_state: list
            List of system state components to monitor.

        score_fn: callable
            Function that is used to calculate each user's predicted scores for
            each candidate item. The score function should take as input
            user_profiles and item_attributes.

        verbose: bool (optional, default: False)
            If True, it enables verbose mode.

        seed: int, optional
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
            An array representing real user preferences. Shape should be
            :math:`|U| \\times |A|`, and should match items.

        items: :class:`~components.items.Items`
            An array representing actual item attributes. Shape should be
            :math:`|A| \\times |I|`, and should match users.

        predicted_scores: :class:`~components.users.PredictedScores`
            An array representing the user preferences as perceived by the
            system. The shape is always :math:`|U| \\times |I|`, where
            :math:`|U|` is the number of users in the system and :math:`|I|`
            is the number of items in the system. The scores are calculated with
            the dot product of :attr:`users_hat` and :attr:`items_hat`.

        num_users: int
            The number of users in the system.

        num_items: int
            The number of items in the system.

        num_items_per_iter: int or str
            Number of items presented to the user per iteration. If `"all"`, then
            the system will serve recommendations from the set of all items in the
            system.

        probabilistic_recommendations: bool (optional, default: False)
            When this flag is set to ``True``, the recommendations (excluding
            any random interleaving) will be randomized, meaning that items
            will be recommended with a probability proportionate to their
            predicted score, rather than the top `k` items, as ranked by their
            predicted score, being recommended.

        random_state: :class:`trecs.random.generators.Generator`

        indices: :obj:`numpy.ndarray`
            A :math:`|U| \\times |I|` array representing the past interactions of each
            user. This keeps track of which items each user has interacted
            with, so that it won't be presented to the user again if
            `repeated_items` are not allowed.

        items_shown: :obj:`numpy.ndarray`
            A :math:`|U| \\times \\text{num_items_per_iter}` array representing the
            indices of the items that each user was shown (i.e., their recommendations)
            from the most recent timestep.

        interactions: :obj:`numpy.ndarray`
            A :math:`|U| \\times 1` array representing the indices of the items
            that each user interacted with at the most recent time step.

        score_fn: callable
            Function that is used to calculate each user's predicted scores for
            each candidate item. The score function should take as input
            ``user_profiles`` and ``item_attributes``.

        interleaving_fn: callable
            Function that is used to determine the indices of items that will be
            interleaved into the recommender system's recommendations. The
            interleaving function should take as input an integer ``k`` (representing
            the number of items to be interleaved in every recommendation set) and
            a matrix ``item_indices`` (representing which items are eligible to be
            interleaved). The function should return a :math:`|U|\\times k` matrix
            representing the interleaved items for each user.
    """

    # pylint: disable=too-many-instance-attributes,too-many-public-methods
    # The recommender system model contains everything needed to run the simulation,
    # so many instance attributes are justified in this case.
    @abstractmethod
    def __init__(  # pylint: disable=R0913,R0912,R0915
        self,
        users_hat,
        items_hat,
        users,
        items,
        num_users,
        num_items,
        num_items_per_iter,
        creators=None,
        probabilistic_recommendations=False,
        measurements=None,
        record_base_state=False,
        system_state=None,
        score_fn=mo.inner_product,
        interleaving_fn=None,
        verbose=False,
        seed=None,
    ):
        # Init logger
        VerboseMode.__init__(self, __name__.upper(), verbose)
        # Initialize measurements
        MeasurementModule.__init__(self)
        # init the recommender system's internal representation of users
        # and items
        self.users_hat = PredictedUserProfiles(users_hat)
        self.items_hat = PredictedItems(items_hat)
        if not callable(score_fn):
            # score function must be a function
            raise TypeError("Custom score function must be a callable method")
        self.score_fn = score_fn
        if interleaving_fn and not callable(interleaving_fn):
            # interleaving function must be callable
            raise TypeError("Custom interleaving function must be a callable method")
        self.interleaving_fn = interleaving_fn
        # set predicted scores
        self.predicted_scores = None
        self.train()
        # determine whether recommendations should be randomized, rather than
        # top-k by predicted score
        self.probabilistic_recommendations = probabilistic_recommendations

        # these variables hold the indices of the items users were shown
        # and the indices of the items users interacted with at the most
        # recent timestep
        self.items_shown = np.empty((num_users, 0))
        self.interactions = np.empty((num_users, 0))

        if not is_valid_or_none(num_users, int):
            raise TypeError("num_users must be an int")
        if not is_valid_or_none(num_items, int):
            raise TypeError("num_items must be an int")
        if not is_valid_or_none(num_items_per_iter, (str, int)):
            raise TypeError("num_items_per_iter must be an int or string 'all'")
        if isinstance(num_items_per_iter, int) and num_items_per_iter < 1:
            # check number of items per iteration is positive
            raise ValueError("num_items_per_iter must be greater than zero")
        if not hasattr(self, "metrics"):
            raise ValueError("You must define at least one measurement module")

        # check users array
        if not is_valid_or_none(users, (list, np.ndarray, sp.spmatrix, Users)):
            raise TypeError("users must be array_like or Users")
        if users is None:
            shape = (self.users_hat.num_users, self.users_hat.num_attrs)
            self.users = Users(size=shape, num_users=num_users, seed=seed)
        if isinstance(users, (list, np.ndarray, sp.spmatrix)):
            # assume that's what passed in is the user's profiles
            self.users = Users(actual_user_profiles=users, num_users=num_users)
        if isinstance(users, Users):
            self.users = users

        # check items array
        if not is_valid_or_none(items, (list, np.ndarray, sp.spmatrix, Items)):
            raise TypeError("items must be array_like or Items")
        if items is None:
            raise ValueError("true item attributes can't be None")
        if isinstance(items, (list, np.ndarray, sp.spmatrix)):
            self.items = Items(items)
        if isinstance(items, Items):
            self.items = items

        if isinstance(creators, Creators):
            self.creators = creators
        else:
            self.creators = None

        # system state
        SystemStateModule.__init__(self)
        if record_base_state:
            self.add_state_variable(
                self.users_hat,
                self.users,
                self.items_hat,
                self.predicted_scores,
            )
            if self.creators is not None:
                self.add_state_variable(self.creators)
        if system_state is not None:
            self.add_state_variable(*system_state)

        self.initialize_user_scores()
        self.num_users = num_users
        self.num_items = num_items
        self.set_num_items_per_iter(num_items_per_iter)
        self.random_state = Generator(seed)
        # Matrix keeping track of the items consumed by each user
        self.indices = np.tile(np.arange(num_items), (num_users, 1))

        # initial metrics measurements (done at the end
        # when the rest of the initial state has been initialized)
        if measurements is not None:
            self.add_metrics(*measurements)

        if self.is_verbose():
            self.log("Recommender system ready")
            self.log(f"Num items: {self.num_items}")
            self.log(f"Users: {self.num_users}")
            self.log(f"Items per iter: {self.num_items_per_iter}")
            if seed is not None:
                self.log(f"Set seed to {seed}")
            else:
                self.log("Seed was not set.")

    @property
    def predicted_user_profiles(self):
        """
        Property that is an alias for the matrix representation of
        predicted user profiles. Returns a matrix of dimension
        :math:`|U|\\times|\\hat{A}|`, where :math:`|\\hat{A}|` is the
        number of attributes that the algorithm uses to represent each
        item and user.
        """
        return self.users_hat.value

    @property
    def predicted_item_attributes(self):
        """
        Property that is an alias for the matrix representation of
        predicted item attributes. Returns a matrix of dimension
        :math:`|\\hat{A}|\\times|I|`,  where :math:`|\\hat{A}|` is the
        number of attributes that the algorithm uses to represent each
        item and user.
        """
        return self.items_hat.value

    @property
    def actual_user_profiles(self):
        """
        Property that is an alias for the matrix representation of
        true user profiles. Returns a matrix of dimension
        :math:`|U|\\times|A^*|`, where :math:`|A^*|` is the number
        of attributes the "true" item/user representation has.
        """
        return self.users.actual_user_profiles.value

    @property
    def actual_item_attributes(self):
        """
        Property that is an alias for the matrix representation of
        actual item attributes. Returns a matrix of dimension
        :math:`|A^*|\\times|I|`, where :math:`|A^*|` is the number
        of attributes the "true" item representation has.
        """
        return self.items.value

    @property
    def actual_user_item_scores(self):
        """
        Property that is an alias for the matrix representation of
        the true user-item score matrix. Returns a matrix of
        dimension :math:`|U|\\times|I|`.
        """
        return self.users.actual_user_scores.value

    @property
    def predicted_user_item_scores(self):
        """
        Property that is an alias for the matrix representation of
        the RS algorithm's predicted user-item score matrix.  Returns
        a matrix of dimension :math:`|U|\\times|I|`.
        """
        return self.predicted_scores.value

    def initialize_user_scores(self):
        """
        If the Users object does not already have known user-item scores,
        then we calculate these scores.
        """
        # users compute their own scores using the true item attributes,
        # unless their own scores are already known to them
        if self.users.get_actual_user_scores() is None:
            self.users.compute_user_scores(self.actual_item_attributes)

    def train(self):
        """
        Updates scores predicted by the system based on the internal state of the
        recommender system. Under default initialization, it updates
        :attr:`predicted_scores` with a dot product of user and item attributes.

        Returns
        --------
            predicted_scores: :class:`~components.users.PredictedScores`
        """
        predicted_scores = self.score_fn(
            self.predicted_user_profiles, self.predicted_item_attributes
        )
        if self.is_verbose():
            self.log(
                "System updates predicted scores given by users (rows) "
                "to items (columns):\n"
                f"{str(predicted_scores)}"
            )
        if self.predicted_scores is None:
            self.predicted_scores = PredictedScores(predicted_scores)
        else:
            self.predicted_scores.value = predicted_scores

    def generate_recommendations(self, k=1, item_indices=None):
        """
        Generate recommendations for each user.

        Parameters
        -----------

            k : int, default 1
                Number of items to recommend.

            item_indices : :obj:`numpy.ndarray`, optional
                A matrix containing the indices of the items each user has not yet
                interacted with. It is used to ensure that the user is presented
                with items they have not already interacted with. If `None`,
                then the user may be recommended items that they have already
                interacted with.

        Returns
        ---------
            Recommendations: :obj:`numpy.ndarray`
        """
        if item_indices is not None:
            if item_indices.size < self.num_users:
                raise ValueError(
                    "At least one user has interacted with all items!"
                    "To avoid this problem, you may want to allow repeated items."
                )
            if k > item_indices.shape[1]:
                raise ValueError(
                    f"There are not enough items left to recommend {k} items to each user."
                )
        if k == 0:
            return np.array([]).reshape((self.num_users, 0)).astype(int)
        # convert to dense because scipy does not yet support argsort - consider
        # implementing our own fast sparse version? see
        # https://stackoverflow.com/questions/31790819/scipy-sparse-csr
        # -matrix-how-to-get-top-ten-values-and-indices
        s_filtered = mo.to_dense(self.predicted_scores.filter_by_index(item_indices))
        row = np.repeat(self.users.user_vector, item_indices.shape[1])
        row = row.reshape((self.num_users, -1))
        if self.probabilistic_recommendations:
            permutation = s_filtered.argsort()
            rec = item_indices[row, permutation]
            # the recommended items will not be exactly determined by
            # predicted score; instead, we will sample from the sorted list
            # such that higher-preference items get more probability mass
            num_items_unseen = rec.shape[1]  # number of items unseen per user
            probabilities = np.logspace(0.0, num_items_unseen / 10.0, num=num_items_unseen, base=2)
            probabilities = probabilities / probabilities.sum()
            picks = np.random.choice(num_items_unseen, k, replace=False, p=probabilities)
            return rec[:, picks]
        else:
            # returns top k indices, sorted from greatest to smallest
            sort_top_k = mo.top_k_indices(s_filtered, k, self.random_state)
            # convert top k indices into actual item IDs
            rec = item_indices[row[:, :k], sort_top_k]
            if self.is_verbose():
                self.log(f"Item indices:\n{str(item_indices)}")
                self.log(
                    f"Top-k items ordered by preference (high to low) for each user:\n{str(rec)}"
                )
            return rec

    def choose_interleaved_items(self, k, item_indices):
        """
        Chooses k items out of the item set to "interleave" into
        the system's recommendations. In this case, we define "interleaving"
        as a process by which items can be inserted into the set of items
        shown to the user, in addition to the recommended items that
        maximize the predicted score. For example, users may want to insert
        random interleaved items to increase the "exploration" of the
        recommender system, or may want to ensure that new items are always
        interleaved into the item set shown to users.
        **NOTE**: Currently, there
        is no guarantee that items that are interleaved are distinct
        from the recommended items. We do guarantee that within the
        set of items interleaved for a particular user, there are no
        repeats.

        Parameters
        -----------

            k : int
                Number of items that should be interleaved in the
                recommendation set for each user.

            item_indices : :obj:`numpy.ndarray`
                Array that contains the valid item indices for each user;
                that is, the indices of items that they have not yet
                interacted with.

        Returns
        ---------
            interleaved_items: :obj:`numpy.ndarray`
        """
        if self.interleaving_fn:
            return self.interleaving_fn(k, item_indices)

        if k == 0:
            return np.array([]).reshape((self.num_users, 0)).astype(int)

        # NOTE: there is currently no guarantee that randomly interleaved items do
        # not overlap with recommended items, since we do not have visibility
        # into the recommended set of items. we do guarantee that for every user,
        # items will not be repeated within the set of interleaved items.
        rand_item = self.random_state.random(item_indices.shape)
        top_k = rand_item.argpartition(-k)[:, -k:]
        row = np.repeat(self.users.user_vector, k).reshape((self.num_users, -1))
        sort_top_k = rand_item[row, top_k].argsort()
        interleaved_items = item_indices[row, top_k[row, sort_top_k]]
        return interleaved_items

    def recommend(
        self,
        startup=False,
        random_items_per_iter=0,
        vary_random_items_per_iter=False,
        repeated_items=True,
    ):
        """
        Implements the recommendation process by combining recommendations and
        new (random) items.

        Parameters
        -----------
            startup: bool, default False
                If True, the system is in "startup"  (exploration) mode and
                only presents the user with new randomly chosen items. This is
                done to maximize exploration.

            random_items_per_iter: int, default 0
                Number of per-user item recommendations that should be
                randomly generated. Passing in ``self.num_items_per_iter``
                will result in all recommendations being randomly generated,
                while passing in ``0`` will result in all recommendations
                coming from predicted score.

            vary_random_items_per_iter: bool, default False
                If ``True``, then at each timestep, the # of items that are recommended
                randomly is itself randomly generated between 0 and
                ``random_items_per_iter``, inclusive.

            repeated_items : bool, default True
                If ``True``, repeated items are allowed in the system -- that is,
                users can interact with the same item more than once.

        Returns
        --------
            Items: :obj:`numpy.ndarray`
                New and recommended items in random order.
        """
        if random_items_per_iter > self.num_items_per_iter:
            raise ValueError(
                "Cannot show more random items per iteration than the total number"
                " of items shown per iteration"
            )

        if startup:
            num_new_items = self.num_items_per_iter
            num_recommended = 0
        else:
            num_new_items = random_items_per_iter
            if vary_random_items_per_iter:
                num_new_items = self.random_state.integers(0, random_items_per_iter + 1)
            num_recommended = self.num_items_per_iter - num_new_items

        item_indices = self.indices
        if not repeated_items:
            # for each user, eliminate items that have been interacted with
            item_indices = item_indices[np.where(item_indices >= 0)]
            item_indices = item_indices.reshape((self.num_users, -1))

        recommended = self.generate_recommendations(k=num_recommended, item_indices=item_indices)

        if self.is_verbose():
            self.log(f"Choice among {item_indices.shape[0]} items")
            if item_indices.shape[1] < num_new_items:
                self.log("Insufficient number of items left!")

        interleaved_items = self.choose_interleaved_items(num_new_items, item_indices)

        if num_new_items > 0:
            items = self.random_state.random((self.num_users, self.num_items_per_iter))
            interleave_mask = np.zeros(items.shape).astype(bool)
            rand_col_idxs = items.argpartition(-num_new_items)[:, -num_new_items:]
            np.put_along_axis(interleave_mask, rand_col_idxs, True, axis=1)
            items[interleave_mask] = interleaved_items.flatten()
            items[~interleave_mask] = recommended.flatten()
            items = items.astype(int)
        else:
            items = recommended

        if self.is_verbose():
            self.log("System picked these items (cols) for each user (rows):\n" + str(items))
        return items

    @abstractmethod
    def _update_internal_state(self, interactions):
        """
        Updates user profiles based on last interaction.

        It must be defined in the concrete class.
        """

    def process_new_items(self, new_items):  # pylint: disable=R0201
        """
        Creates new item representations based on items that were just created.

        Must be defined in the concrete class.
        """
        raise RuntimeError(
            "process_new_items not defined. Support for representing new"
            "items must be implemented by the user!"
        )

    def process_new_users(self, new_users, **kwargs):  # pylint: disable=R0201
        """
        Creates new user representations based on items that were just created.

        Must be defined in the concrete class.
        """
        raise RuntimeError(
            "process_new_users not defined. Support for representing new"
            "users must be implemented by the user!"
        )

    def run(
        self,
        timesteps=50,
        startup=False,
        train_between_steps=True,
        random_items_per_iter=0,
        vary_random_items_per_iter=False,
        repeated_items=True,
        no_new_items=False,
        disable_tqdm=False,
    ):  # pylint: disable=too-many-arguments
        """
        Runs simulation for the given timesteps.

        Parameters
        -----------

            timestep : int, default 50
                Number of timesteps for simulation.

            startup : bool, default False
                If ``True``, it runs the simulation in startup mode (see
                :func:`recommend` and :func:`startup_and_train`)

            train_between_steps : bool, default True
                If ``True``, the model is retrained after each timestep with the
                information gathered in the previous step.

            random_items_per_iter: int, default 0
                Number of per-user item recommendations that should be
                randomly generated. Passing in ``self.num_items_per_iter`` will
                result in all recommendations being randomly generated, while passing
                in ``0`` will result in all recommendations coming from predicted scores.

            vary_random_items_per_iter: bool, default False
                If ``True``, then at each timestep, the # of items that are recommended
                randomly is itself randomly generated between 0 and
                ``random_items_per_iter``, inclusive.

            repeated_items : bool, default True
                If ``True``, repeated items are allowed in the system -- that is,
                the system can recommend items to users that they've already previously
                interacted with.

            no_new_items : bool, default False
                If ``True``, then no new items are created during these timesteps. This
                can be helpful, say, during a "training" period where no new items should be
                made.
        """
        if len(self.metrics) == 0:  # warn user if no measurements are defined
            error_msg = (
                "No measurements are currently defined for the simulation. Please add "
                "measurements if desired."
            )
            warnings.warn(error_msg)
        if not startup and self.is_verbose():
            self.log("Running recommendation simulation using recommendation algorithm...")
        for timestep in tqdm(range(timesteps), disable=disable_tqdm):
            if self.is_verbose():
                self.log(f"Step {timestep}")
            if self.creators is not None and not no_new_items:
                self.create_and_process_items()
                if self.expand_items_per_iter:
                    # expand set of items recommended per iteration
                    self.set_num_items_per_iter("all")
            self.items_shown = self.recommend(
                startup=startup,
                random_items_per_iter=random_items_per_iter,
                vary_random_items_per_iter=vary_random_items_per_iter,
                repeated_items=repeated_items,
            )
            self.interactions = self.users.get_user_feedback(self.items_shown)
            if not repeated_items:
                self.indices[self.users.user_vector, self.interactions] = -1
            self._update_internal_state(self.interactions)
            if self.is_verbose():
                self.log("Recorded user interaction:\n" + str(self.interactions))
                self.log(
                    "System updates user profiles based on last interaction:\n"
                    + str(self.users_hat)
                )
            # update creators if any
            if self.creators is not None:
                self.creators.update_profiles(self.interactions, self.actual_item_attributes)
            # update users if needed
            if self.users.drift > 0:
                # update user profiles based on the attributes of items they
                # interacted with
                interact_attrs = self.actual_item_attributes.T[self.interactions, :]
                self.users.update_profiles(interact_attrs)
                # update user scores
                self.users.compute_user_scores(self.actual_item_attributes)
            # train between steps:
            if train_between_steps:
                self.train()
            # record state and compute metrics
            self.record_state()
            self.measure_content()

    def startup_and_train(self, timesteps=50, no_new_items=False):
        """
        Runs simulation in startup mode by calling :func:`run` with
        startup=True. For more information about startup mode, see :func:`run`
        and :func:`recommend`.

        Parameters
        -----------

            timesteps : int, default 50
                Number of timesteps for simulation

            no_new_items : bool, default False
                If ``True``, then no new items are created during these timesteps.
                This is only relevant when you have item
                :class:`~components.creators.Creators`. This can be helpful, say, during
                a "training" period where no new items should be made.
        """
        if self.is_verbose():
            self.log("Startup -- recommend random items")
        self.run(timesteps, startup=True, train_between_steps=False, no_new_items=no_new_items)
        self.train()

    def create_and_process_items(self):
        """
        Creates and processes items made by content creators
        """
        # generate new items
        new_items = self.creators.generate_items()  # should be A x I
        self.num_items += new_items.shape[1]  # increment number of items
        # concatenate old items with new items
        self.items.append_new_items(new_items)
        # generate new internal system representations of the items
        new_items_hat = self.process_new_items(new_items)
        self.items_hat.append_new_items(new_items_hat)

        self.add_new_item_indices(new_items.shape[1])
        # create new predicted scores if not in startup
        new_item_pred_score = self.score_fn(self.users_hat.value, new_items_hat)
        self.predicted_scores.append_item_scores(new_item_pred_score)
        # have users update their own scores too
        self.users.score_new_items(new_items)

    def add_users(self, new_users, **kwargs):
        """
        Create pool of new users

        Parameters
        -----------

            new_users: :obj:`numpy.ndarray`
                An array representing users. Should be of dimension
                :math:`|U_n| \\times |A|`, where :math:`|U_n|` represents
                the number of new users, and :math:`|A|` represents
                the number of attributes for each user profile.

            **kwargs:
                Any additional information about users
                can be passed through `kwargs` (see `social.py`) for
                an example.
        """
        self.num_users += new_users.shape[0]
        # register new user profiles & new user-item scores
        self.users.append_new_users(new_users, self.items.value)

        # update predicted user profiles
        new_users_hat = self.process_new_users(new_users, **kwargs)
        self.users_hat.append_new_users(new_users_hat)

        self.add_new_user_indices(new_users.shape[0])
        # create new predicted scores if not in startup
        new_item_pred_score = self.score_fn(new_users_hat, self.items_hat.value)
        self.predicted_scores.append_user_scores(new_item_pred_score)

    def set_num_items_per_iter(self, num_items_per_iter):
        """Change the number of items that will be shown
        to each user per iteration.
        """
        if num_items_per_iter == "all":
            self.num_items_per_iter = self.num_items
            self.expand_items_per_iter = True
        else:
            self.expand_items_per_iter = False
            self.num_items_per_iter = num_items_per_iter

    def add_new_item_indices(self, num_new_items):
        """
        Expands the indices matrix to include entries for new items that
        were created.

        Parameters
        -----------
            num_new_items (int): The number of new items added to the system
            in this iteration
        """
        num_existing_items = self.indices.shape[1]
        new_indices = num_existing_items + np.tile(np.arange(num_new_items), (self.num_users, 1))
        self.indices = np.hstack([self.indices, new_indices])

    def add_new_user_indices(self, num_new_users):
        """
        Expands the indices matrix to include entries for new users that
        were created.

        Parameters
        -----------
            num_new_users (int): The number of new items added to the system
            in this iteration
        """
        new_indices = np.tile(np.arange(self.num_items), (num_new_users, 1))
        self.indices = np.vstack([self.indices, new_indices])

    def get_measurements(self):
        """
        Returns all available measurements. For more details, please see the
        :class:`~metrics.measurement.Measurement` class.

        Returns
        --------
        Monitored measurements: dict
        """
        if len(self.metrics) < 1:
            return None
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
        :attr:`~base.base_components.BaseComponent.state_history` of the
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
