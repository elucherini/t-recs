import numpy as np
from abc import ABCMeta, abstractmethod
from .measurement import MSEMeasurement
from .users import Users
from .utils import normalize_matrix, is_valid_or_none
from .debug import VerboseMode
from .items import Items

class Recommender(VerboseMode, metaclass=ABCMeta):
    """Abstract class representing a recommender system.

        All attributes and methods in this class are generic to all recommendation systems
        implemented.

        Args:
            user_representation (:obj:`numpy.ndarray`): An array representing users. The
                shape and meaning depends on the implementation of the concrete class.
            item_representation (:obj:`numpy.ndarray`): An array representing items. The
                shape and meaning depends on the implementation of the concrete class.
            actual_user_representation (:obj:`numpy.ndarray`): An array representing real user
                preferences unknown to the system. The shape and meaning depends on the
                implementation of the concrete class.
            num_users (int): The number of users in the system.
            num_items (int): The number of items in the system.
            num_items_per_iter (int): Number of items presented to the user at each
                iteration.
            num_new_items (int): Number of new items that the systems add if it runs out
                of items that the user can interact with.
            verbose (bool, optional): If True, enables verbose mode. Disabled by default.

        Attributes:
            Attributes inherited by :class:`VerboseMode`, plus:
            user_profiles (:obj:`numpy.ndarray`): An array representing users, matching
                user_representation. The shape and meaning depends on the implementation
                of the concrete class.
            item_attributes (:obj:`numpy.ndarray`): An array representing items, matching
                item_representation. The shape and meaning depends on the implementation
                of the concrete class.
            actual_users (:obj:`numpy.ndarray`): An array representing real user
                preferences, matching actual_users. The shape and meaning depends
                on the implementation of the concrete class.
            predicted_scores (:obj:`numpy.ndarray`): An array representing the user
                preferences as perceived by the system. The shape is always |U|x|I|,
                where |U| is the number of users in the system and |I| is the number of
                items in the system. The scores are calculated with the dot product of
                user_profiles and item_attributes.
            measurements (:class:`Measurements`): Measurement module. See :class:`Measurements`.
            num_users (int): The number of users in the system.
            num_items (int): The number of items in the system.
            num_items_per_iter (int): Number of items presented to the user per iteration.
            num_new_items (int): Number of new items that the systems add if it runs out
                of items that the user can interact with.
            indices (:obj:`numpy.ndarray`): A |U|x|I| array representing the past
                interactions of each user. This keeps track of which items each user
                has interacted with, so that it won't be presented to the user again.
            user_vector (:obj:`numpy.ndarray`): An array of length |U| s.t. user_vector_u = u
                for u in U.
            item_vector (:obj:`numpy.ndarray`): An array of length |I| s.t. item_vector_i = i
                for i in I.
    """
    @abstractmethod
    def __init__(self, user_representation, item_representation,
                 actual_user_representation, num_users, num_items,
                 num_items_per_iter, num_new_items, verbose=False):
        # Init logger
        VerboseMode.__init__(self, __name__.upper(), verbose)
        # init users and items
        self.user_profiles = user_representation
        self.item_attributes = Items(item_representation)
        # set predicted scores
        self.predicted_scores = self.train(self.user_profiles, self.item_attributes,
                                           normalize=True)
        assert(self.predicted_scores is not None)

        if actual_user_representation is not None:
            self.actual_users = actual_user_representation
        else:
            self.actual_users = Users(size=self.user_profiles.shape)

        self.actual_users.compute_user_scores(self.train)

        if not is_valid_or_none(num_users, int):
            raise ValueError("num_users must be an int")
        if not is_valid_or_none(num_items, int):
            raise ValueError("num_items must be an int")
        if not is_valid_or_none(num_items_per_iter, int):
            raise ValueError("num_items_per_iter must be an int")
        if not is_valid_or_none(num_new_items, int):
            raise ValueError("num_new_items must be an int")
        self.num_users = num_users
        self.num_items = num_items
        self.num_items_per_iter = num_items_per_iter
        self.num_new_items = num_new_items
        # Matrix keeping track of the items consumed by each user
        self.indices = np.tile(np.arange(num_items), (num_users, 1))
        self.log('Recommender system ready')
        self.user_vector = np.arange(num_users, dtype=int)
        self.item_vector = np.arange(2, num_items_per_iter + 2, dtype=int)
        self.log('Num items: %d' % self.num_items)
        self.log('Users: %d' % self.num_users)
        self.log('Items per iter: %d' % self.num_items_per_iter)


    def train(self, user_profiles, item_attributes, normalize=True):
        """ Updates recommender based on past interactions for better user predictions.

            Args:
                normalize (bool, optional): set to True if the scores should be normalized,
                    False otherwise.
        """
        if normalize:
            user_profiles = normalize_matrix(user_profiles, axis=1)
        predicted_scores = np.dot(user_profiles, item_attributes)
        self.log('System updates predicted scores given by users (rows) ' + \
            'to items (columns):\n' + str(predicted_scores))
        assert(predicted_scores is not None)
        return predicted_scores

    def generate_recommendations(self, k=1, indices_prime=None):
        """ Generate recommendations

            Args:
                k (int, optional): number of items to recommend.
                indices_prime (:obj:numpy.ndarray, optional): a matrix containing the
                    indices of the items each user has interacted with. It is used to
                    ensure that the user is presented with items they have already
                    interacted with.

            Returns:
                An array of k recommendations.

            Todo:
                * Group matrix manipulations into util functions
        """
        if indices_prime is None:
            indices_prime = self.indices[np.where(self.indices>=0)]
            indices_prime = indices_prime.reshape((self.num_users, -1))
        if indices_prime.size == 0 or k > indices_prime.shape[1]:
            self.log('Insufficient number of items left!')
            #self._expand_items()
            indices_prime = self.indices[np.where(self.indices>=0)]
            indices_prime = indices_prime.reshape((self.num_users, -1))
        row = np.repeat(self.user_vector, indices_prime.shape[1])
        row = row.reshape((self.num_users, -1))
        #self.log('row:\n' + str(row))
        self.log('Row:\n' + str(row))
        self.log('Indices_prime:\n' + str(indices_prime))
        s_filtered = self.predicted_scores[row, indices_prime]
        #self.log('s_filtered\n' + str(s_filtered))
        permutation = s_filtered.argsort()
        #self.log('permutation\n' + str(permutation))
        rec = indices_prime[row, permutation]
        probabilities = np.logspace(0.0, rec.shape[1]/10.0, num=rec.shape[1], base=2)
        probabilities = probabilities/probabilities.sum()
        self.log('Items ordered by preference for each user:\n' + str(rec))
        picks = np.random.choice(permutation.shape[1], p=probabilities, size=(self.num_users, k))
        #self.log('recommendations\n' + str(rec[np.repeat(self.user_vector, k).reshape((self.num_users, -1)), picks]))
        #print(self.predicted_scores.argsort()[:,::-1][:,0:5])
        return rec[np.repeat(self.user_vector, k).reshape((self.num_users, -1)), picks]
        #return self.predicted_scores.argsort()[:,::-1][:,0:k]

    def recommend(self, startup=False):
        """Implements the recommendation process by combining recommendations and
            new (random) items.

            Args:
                startup (bool, optional): If True, the system is in "startup" (exploration) mode
                    and only presents the user with new randomly-chosen items. This is to maximize
                    exploration.

            Returns:
                New and recommended items in random order.
        """
        if startup:
            num_new_items = self.num_items_per_iter
            num_recommended = 0
        else:
            num_new_items = np.random.randint(0, self.num_items_per_iter)
            num_recommended = self.num_items_per_iter - num_new_items

        if num_recommended == 0 and num_new_items == 0:
            # TODO throw exception here
            print("Nope")
            return

        if num_recommended > 0:
            recommended = self.generate_recommendations(k=num_recommended)
            assert(num_recommended == recommended.shape[1])
            assert(recommended.shape[0] == self.num_users)
            self.log('System recommended these items (cols) to each user ' +\
                '(rows):\n' + str(recommended))
        else:
            recommended = None
        indices_prime = self.indices[np.where(self.indices>=0)]
        indices_prime = indices_prime.reshape((self.num_users, -1))
        # Current assumptions:
        # 1. Interleave new items and recommended items
        # 2. Each user interacts with one element depending on preference
        # 3. Users can't interact with the same item more than once
        assert(np.count_nonzero(self.indices == -1) % self.num_users == 0)
        self.log("Choice among %d items" % (indices_prime.shape[0]))
        if indices_prime.shape[1] < num_new_items:
            self.log('Insufficient number of items left!')
            #self._expand_items()
            indices_prime = self.indices[np.where(self.indices>=0)]
            indices_prime = indices_prime.reshape((self.num_users, -1))

        if num_new_items:
            col = np.random.randint(indices_prime.shape[1], size=(self.num_users, num_new_items))
            row = np.repeat(self.user_vector, num_new_items).reshape((self.num_users, -1))
            new_items = indices_prime[row, col]
            self.log('System picked these items (cols) randomly for each user ' + \
                '(rows):\n' + str(new_items))

        if num_recommended and num_new_items:
            items = np.concatenate((recommended, new_items), axis=1)
        elif num_new_items:
            items = new_items
        else:
            items = recommended
        np.random.shuffle(items.T)
        return items

    @abstractmethod
    def _update_user_profiles(self):
        """ Updates user profiles based on last interaction.

            It must be defined in the concrete class.
        """
        pass


    def run(self, timesteps=50, startup=False, train_between_steps=True,
            repeated_items=False):
        """ Runs simulation for the given timesteps.

            Args:
                timestep (int, optional): number of timesteps for simulation
                startup (bool, optional): if True, it runs the simulation in
                    startup mode (see recommend() and startup_and_train())
                train_between_steps (bool, optional): if True, the model is
                    retrained after each step with the information gathered
                    in the previous step.
                repeated_items (bool, optional): if True, repeated items are allowed
                    in the system -- that is, users can interact with the same
                    item more than once. Examples of common instances in which
                    this is useful: infection and network propagation models.
                    Default is False.
        """
        if not startup:
            self.log('Run -- interleave recommendations and random items ' + \
                'from now on')
        for t in range(timesteps):
            self.log('Step %d' % t)
            items = self.recommend(startup=startup)
            interactions = self.actual_users.get_user_feedback(items,
                                                        self.user_vector)
            if not repeated_items:
                self.indices[self.user_vector, interactions] = -1
            self._update_user_profiles(interactions)
            self.log("System updates user profiles based on last interaction:\n" + \
                str(self.user_profiles.astype(int)))
            self.measure_content(interactions, step=t)
            #self.get_user_feedback()
            # train between steps:
            if train_between_steps:
                self.predicted_scores = self.train(self.user_profiles,
                                                   self.item_attributes)
        # If no training in between steps, train at the end:
        if not train_between_steps:
            self.predicted_scores = self.train(self.user_profiles,
                                               self.item_attributes)


    def startup_and_train(self, timesteps=50):
        """ Runs simulation in startup mode by calling run() with startup=True.
            For more information about startup mode, see run() and recommend().

            Args:
                timestep (int, optional): number of timesteps for simulation
        """
        self.log('Startup -- recommend random items')
        return self.run(timesteps, startup=True, train_between_steps=False)

    def _expand_items(self, num_new_items=None):
        """ Increases number of items in the system.

            Args:
                num_new_items (int, optional): number of new items to add to the system.
                    If None, it is equal to twice the number of items presented to the user
                    in one iteration.
        """
        '''
        if num_new_items is None:
            num_new_items = 2 * self.num_items_per_iter
        if not isinstance(num_new_items, int):
            raise TypeError("num_new_items should be int, is instead %s" % (
                                                                    type(num_new_items)))
        if num_new_items < 1:
            raise ValueError("Can't increment items by a number smaller than 1!")
        #new_indices = np.tile(self.item_attributes.expand_items(self, num_new_items),
        #    (self.num_users,1))
        self.indices = np.concatenate((self.indices, new_indices), axis=1)
        self.actual_users.compute_user_scores(self.train)
        self.predicted_scores = self.train(self.user_profiles, self.item_attributes)
        '''
        pass

    def get_measurements(self):
        """ Returns all available measurements. For more details,
            please see the :class:`Measurements` class.

            Returns: Pandas dataframe of all available measurements.
        """
        measurements = {'Timesteps': None}
        for metric in self.measurements:

            measurements = {**measurements, **metric.get_measurement()}
        #for name, measure in measurements.items():
            #self.debugger.pyplot_plot(measure['x'], measure['y'],
            #    title=str(name.capitalize()), xlabel='Timestep',
            #    ylabel=str(name))
        if measurements['Timesteps'] == None:
            # pick first measurement's length for # of timesteps since they're going to be the same
            elapsed = np.arange(self.measurements[0].get_timesteps())
            measurements['Timesteps'] = elapsed
        return measurements

    def measure_content(self, interactions, step):
        """ Calls method in the :class:`Measurements` module to record metrics.
            For more details, see the :class:`Measurements` class and its measure
            method.

            Args:
                interactions (:obj:`numpy.ndarray`): matrix of interactions
                    per users at a given time step.
                step (int): step on which the recorded interactions refers to.
        """
        for metric in self.measurements:
            metric.measure(step, interactions, self)
