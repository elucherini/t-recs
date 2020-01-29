import numpy as np
from .recommender import Recommender
from .debug import VerboseMode
from .utils import normalize_matrix

class ContentFiltering(Recommender, VerboseMode):
    def __init__(self, num_users=100, num_items=1250, num_attributes=None,
        item_representation=None, user_representation=None, verbose=False,
        num_items_per_iter=10, num_new_items=30):
        # Init logger
        VerboseMode.__init__(self, __name__.upper(), verbose)
        self.binary = True
        # Give precedence to item_representation, otherwise build random one
        if item_representation is not None:
            self.item_attributes = item_representation
            num_attributes = item_representation.shape[0]
            num_items = item_representation.shape[1]
        else:
            if num_attributes is None:
                if user_representation is not None:
                    num_attributes = user_representation.shape[1]
                else:
                    num_attributes = np.random.randint(2, max(3, int(num_items - num_items / 10)))
            self.item_attributes = self._init_random_item_attributes(num_attributes, 
                num_items, binary=self.binary)

        assert(num_attributes is not None)
        assert(self.item_attributes is not None)
        # Give precedence to user_representation, otherwise build random one
        if user_representation is None:
            self.user_profiles = np.zeros((num_users, num_attributes), dtype=int)
        elif user_representation.shape[1] == self.item_attributes.shape[0]:
            self.user_profiles = user_representation
            num_users = user_representation.shape[0]
        else:
            raise Exception("It should be user_representation.shape[1] == item_representation.shape[0]")
        
        # Initialize recommender system
        Recommender.__init__(self, num_users, num_items, num_items_per_iter, num_new_items)
        #self.log('Type of recommendation system: %s' % __name__)
        #self.log('Num attributes: %d' % self.item_attributes.shape[0])
        #self.log('Attributes of each item (rows):\n%s' % \
        #    (str(self.item_attributes.T)))

    def _init_random_item_attributes(self, num_attributes, num_items, binary=False):
        # TODO: attributes from distributions
        if binary:
            dist = np.random.binomial(1, .3, size=(num_items, num_attributes))
        else: # 
            dist = np.random.random(size=(num_items, num_attributes))
            dist = dist / dist.sum(axis=1)[:,None]
        return dist.T

    def _store_interaction(self, interactions):
        interactions_per_user = np.zeros((self.num_users, self.num_items))
        interactions_per_user[self.user_vector, interactions] = 1
        user_attributes = np.dot(interactions_per_user, self.item_attributes.T)
        self.user_profiles = np.add(self.user_profiles, user_attributes)

    def _expand_items(self, num_new_items=None):
        if isinstance(num_new_items, int):
            num_new_items = 2 * self.num_items_per_iter
        new_indices = np.tile(self.item_attributes.expand_items(self, num_new_items),
            (self.num_users,1))
        self.indices = np.concatenate((self.indices, new_indices), axis=1)
        self.actual_user_scores.expand_items(self.item_attributes)
        self.train()

    def train(self, normalize=True):
        # Normalize user_profiles
        assert(self.user_profiles.shape[1] == self.item_attributes.shape[0])
        if normalize:
            user_profiles = normalize_matrix(self.user_profiles, axis=1)
        else:
            user_profiles = self.user_profiles
        Recommender.train(self, user_profiles=user_profiles)

    def interact(self, step=None, startup=False):
        if startup:
            num_new_items = self.num_items_per_iter
            num_recommended = 0
        else:
            num_new_items = np.random.randint(0, self.num_items_per_iter)
            num_recommended = self.num_items_per_iter-num_new_items
        '''
        else:
            # TODO: these may be constants or iterators on vectors
            num_new_items = self.num_new_items
            num_recommended = self.num_recommended
        '''
        interactions = Recommender.interact(self, num_recommended, num_new_items)
        self.measure_content(interactions, step=step)
        self._store_interaction(interactions)
        self.log("System updates user profiles based on last interaction:\n" + \
            str(self.user_profiles.astype(int)))

    def recommend(self, k=1, indices_prime=None):
        return Recommender.recommend(self, k=k, indices_prime=indices_prime)

    def startup_and_train(self, timesteps=50):
        self.log('Startup -- recommend random items')
        return self.run(timesteps, startup=True, train_between_steps=False)

    def run(self, timesteps=50, startup=False, train_between_steps=True):
        if not startup:
            self.log('Run -- interleave recommendations and random items ' + \
                'from now on')
        #self.measurements.set_delta(timesteps)
        for t in range(timesteps):
            self.log('Step %d' % t)
            self.interact(step=t, startup=startup)
            #super().run(startup=False, train=train, step=step)
            if train_between_steps:
                self.train()
        # If no training in between steps, train at the end: 
        if not train_between_steps:
            self.train()
        #return super().run(timesteps, startup=False, train=train)

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