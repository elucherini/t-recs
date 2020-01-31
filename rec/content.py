import numpy as np
from .recommender import Recommender
from .debug import VerboseMode
from .stats import Distribution

class ContentFiltering(Recommender, VerboseMode):
    def __init__(self, num_users=100, num_items=1250, num_attributes=None,
        item_representation=None, user_representation=None, actual_user_preferences=None,
        verbose=False, num_items_per_iter=10, num_new_items=30):
        # Init logger
        VerboseMode.__init__(self, __name__.upper(), verbose)
        # Give precedence to item_representation, otherwise build random one
        if item_representation is not None:
            #self.item_attributes = item_representation
            num_attributes = item_representation.shape[0]
            num_items = item_representation.shape[1]
        else:
            if num_items is None:
                raise ValueError("num_items and item_representation can't be both None")
            if num_attributes is None:
                if user_representation is not None:
                    num_attributes = user_representation.shape[1]
                else:
                    num_attributes = np.random.randint(2, max(3, int(num_items - num_items / 10)))
            item_representation = Distribution(distr_type='binom', p=.3, n=1, 
                                    size=(num_attributes, num_items)).compute()

        assert(num_attributes is not None)
        assert(item_representation is not None)
        # Give precedence to user_representation, otherwise build random one
        if user_representation is None:
            user_representation = np.zeros((num_users, num_attributes), dtype=int)
        elif user_representation.shape[1] == item_representation.shape[0]:
            num_users = user_representation.shape[0]
        else:
            raise ValueError("It should be user_representation.shape[1]" + \
                                " == item_representation.shape[0]")
        assert(user_representation is not None)
        # Initialize recommender system
        Recommender.__init__(self, user_representation, item_representation, actual_user_preferences,
                                num_users, num_items, num_items_per_iter, num_new_items)
        #self.log('Type of recommendation system: %s' % __name__)
        #self.log('Num attributes: %d' % self.item_attributes.shape[0])
        #self.log('Attributes of each item (rows):\n%s' % \
        #    (str(self.item_attributes.T)))

    def update_user_profiles(self, interactions):
        interactions_per_user = np.zeros((self.num_users, self.num_items))
        interactions_per_user[self.user_vector, interactions] = 1
        user_attributes = np.dot(interactions_per_user, self.item_attributes.T)
        self.user_profiles = np.add(self.user_profiles, user_attributes)

    def train(self, normalize=True):
        # Normalize user_profiles
        assert(self.user_profiles.shape[1] == self.item_attributes.shape[0])
        Recommender.train(self, normalize=normalize)

