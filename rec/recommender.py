import numpy as np
from abc import ABCMeta, abstractmethod
from .measurements import Measurements
from .user_scores import ActualUserScores
from .utils import normalize_matrix

# Recommender systems: abstract class
class Recommender(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, user_representation, item_representation, actual_user_preferences,
                            num_users, num_items, num_items_per_iter, num_new_items):
        self.user_profiles = user_representation
        self.item_attributes = item_representation
        # set predicted scores
        self.train(normalize=False)

        if actual_user_preferences is not None:
            self.actual_user_scores = actual_user_preferences
        else:
            self.actual_user_scores = ActualUserScores(num_users=num_users,
                    item_representation=item_representation,
                    normalize=True)

        self.measurements = Measurements()
        self.num_users = num_users
        self.num_items = num_items
        self.num_items_per_iter = num_items_per_iter
        self.num_new_items = num_new_items
        # Matrix keeping track of the items consumed by each user
        self.indices = np.tile(np.arange(num_items), (num_users, 1))
        # NOTE user_preferences either accepts False (randomize user preferences),
        # or it accepts a matrix of user preferences
        self.log('Recommender system ready')
        self.user_vector = np.arange(num_users, dtype=int)
        self.item_vector = np.arange(2, num_items_per_iter + 2, dtype=int)
        self.log('Num items: %d' % self.num_items)
        self.log('Users: %d' % self.num_users)
        self.log('Items per iter: %d' % self.num_items_per_iter)

    # Train recommender system
    def train(self, normalize=True):
        if normalize:
            user_profiles = normalize_matrix(self.user_profiles, axis=1)
        else:
            user_profiles = self.user_profiles
        self.predicted_scores = np.dot(user_profiles, self.item_attributes)
        self.log('System updates predicted scores given by users (rows) ' + \
            'to items (columns):\n' + str(self.predicted_scores))

    # Assume scores two-dimensional
    def generate_recommendations(self, k=1, indices_prime=None):
        if indices_prime is None:
            indices_prime = self.indices[np.where(self.indices>=0)]
            indices_prime = indices_prime.reshape((self.num_users, -1))
        #self.log('Indices_prime:\n' + str(indices_prime))
        if indices_prime.size == 0 or k > indices_prime.shape[1]:
            self.log('Insufficient number of items left!')
            self._expand_items()
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
        assert(np.count_nonzero(self.indices == -1) % self.num_users == 0)
        if indices_prime.shape[1] < num_new_items:
            self.log('Insufficient number of items left!')
            self._expand_items()
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
        def update_user_profiles(self):
            pass


    def run(self, timesteps=50, startup=False, train_between_steps=True):
        if not startup:
            self.log('Run -- interleave recommendations and random items ' + \
                'from now on')
        for t in range(timesteps):
            self.log('Step %d' % t)
            # 
            items = self.recommend(startup=startup)
            interactions = self.actual_user_scores.get_user_feedback(items, 
                                                        self.user_vector)
            self.indices[self.user_vector, interactions] = -1
            self.update_user_profiles(interactions)
            self.log("System updates user profiles based on last interaction:\n" + \
                str(self.user_profiles.astype(int)))
            self.measure_content(interactions, step=t)
            #self.get_user_feedback()
            # train between steps:
            if train_between_steps:
                self.train()
        # If no training in between steps, train at the end:
        if not train_between_steps:
            self.train()


    def startup_and_train(self, timesteps=50):
        self.log('Startup -- recommend random items')
        return self.run(timesteps, startup=True, train_between_steps=False)

    def _expand_items(self, num_new_items=None):
        if not isinstance(num_new_items, int) or num_new_items < 1:
            num_new_items = 2 * self.num_items_per_iter
        new_indices = np.tile(self.item_attributes.expand_items(self, num_new_items),
            (self.num_users,1))
        self.indices = np.concatenate((self.indices, new_indices), axis=1)
        self.actual_user_scores.expand_items(self.item_attributes)
        self.train()


    def get_heterogeneity(self):
        heterogeneity = self.measurements.get_measurements()['delta']
        return heterogeneity

    def get_measurements(self):
        measurements = self.measurements.get_measurements()
        #for name, measure in measurements.items():
            #self.debugger.pyplot_plot(measure['x'], measure['y'],
            #    title=str(name.capitalize()), xlabel='Timestep', 
            #    ylabel=str(name))
        return measurements

    def measure_content(self, interactions, step):
        self.measurements.measure(step, interactions, self.num_users,
            self.num_items, self.predicted_scores,
            self.actual_user_scores.get_actual_user_scores())
