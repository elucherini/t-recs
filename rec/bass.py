from .recommender import Recommender
from .socialgraph import BinarySocialGraph
from .distribution import Generator
from .measurement import StructuralVirality
from .utils import get_first_valid, is_array_valid_or_none, is_equal_dim_or_none, all_none, is_valid_or_none
import numpy as np
import math

class BassModel(BinarySocialGraph, Recommender):
    """Bass model that, for now, only supports one item at a time"""
    def __init__(self, num_users=100, num_items=1, infection_state=None,
        item_representation=None, user_representation=None, infection_threshold=None,
        actual_user_scores=None, verbose=False, num_items_per_iter=10, num_new_items=30):
        # these are not allowed to be None at the same time
        if all_none(user_representation, num_users, infection_state):
            raise ValueError("user_representation, num_users, and infection_state can't be all None")
        if all_none(item_representation, num_items, infection_state):
            raise ValueError("item_representation, num_items, and infection_state can't be all None")
        if not is_array_valid_or_none(user_representation, ndim=2, square=True):
            raise ValueError("user_representation is invalid")
        if not is_array_valid_or_none(infection_state, ndim=2, square=False):
            raise TypeError("infection_state is invalid")

        if not is_equal_dim_or_none(getattr(user_representation, 'shape', [None])[0],
                                  getattr(infection_state, 'shape', [None])[0]):
            raise ValueError("user_representation and infection_state should be of " + \
                             "same size on dimension 0")

        # Determine num_users, give priority to user_representation
        # At the end of this, user_representation and num_users should not be None
        # In the arguments, I either get shape[0], or None if the matrix is None
        num_users = get_first_valid(getattr(user_representation,
                                                     'shape', [None])[0],
                                             getattr(infection_state,
                                                     'shape', [None])[0],
                                             num_users)
        if user_representation is None:
            # TODO SocialGraph module
            user_representation = np.diag(np.diag(np.ones((num_users, num_users),
                                                          dtype=int)))

        assert(num_users is not None)
        assert(user_representation is not None)
        assert(num_users == user_representation.shape[0] == user_representation.shape[1])
        # Determine num_items, give priority to item_representation
        # At the end of this, item_representation should not be None
        if not is_array_valid_or_none(item_representation, ndim=2, square=False):
            raise ValueError("item_representation is invalid")
        if not is_equal_dim_or_none(getattr(item_representation,
                                             'shape', [None, None])[1],
                                  getattr(infection_state,
                                          'shape', [None, None])[1]):
            raise ValueError("item_representation and infection_state should be of" + \
                             "same size on dimension 1")

        # Define number of users based on input
        # In the arguments, I either get shape[0] (or shape[1]),
        #                   or None if the matrix is None
        num_items = get_first_valid(getattr(item_representation,
                                            'shape', [None, None])[1],
                                    getattr(infection_state,
                                            'shape', [None, None])[1],
                                             num_items)
        if item_representation is None:
            item_representation = Generator().uniform(size=(1,num_items))

        assert(num_items is not None)
        assert(item_representation is not None)
        # Define infection_state
        if infection_state is None:
        # TODO change parameters
            infection_state = np.zeros((num_users, num_items))
            random_infections = (np.random.randint(num_users),
                                 np.random.randint(num_items))
            infection_state[random_infections] = 1
        assert(infection_state is not None)
        self.infection_state = infection_state
        # TODO support separate threshold for each user
        if not infection_threshold or infection_threshold >= 1:
            infection_threshold = np.random.random()
        assert(infection_threshold is not None)
        assert(infection_threshold < 1 and infection_threshold > 0)
        self.infection_threshold = abs(infection_threshold)
        self.measurements = [StructuralVirality(np.copy(infection_state))]
        # Initialize recommender system
        num_items_per_iter = 1
        Recommender.__init__(self, user_representation, item_representation, actual_user_scores,
                                num_users, num_items, num_items_per_iter, num_new_items, verbose=verbose)

    def _update_user_profiles(self, interactions):
        """ Private function that updates user profiles with data from
            latest interactions.

            Specifically, this function converts interactions into item attributes.
            For example, if user u has interacted with item i, then the i's attributes
            will be updated to increase the similarity between u and i.

        Args:
            interactions (numpy.ndarray): An array of item indices that users have
                interacted with in the latest step. Namely, interactions_u represents
                the index of the item that the user has interacted with.

        """
        infection_probabilities = self.predicted_scores[self.user_vector, interactions]
        newly_infected = np.where(infection_probabilities > self.infection_threshold)
        if newly_infected[0].shape[0] > 0:
            self.infection_state[newly_infected, interactions[newly_infected]] = 1


    def train(self, user_profiles=None, item_attributes=None, normalize=False):
        """ Overrides train method of parent class :class:`Recommender`.

            Args:
                normalize (bool, optional): set to True if the scores should be normalized,
            False otherwise.
        """
        # normalizing the user profiles is meaningless here
        # This formula comes from Goel et al., The Structural Virality of Online Diffusion
        #print(np.where(np.log(1 - self.item_attributes) == np.nan))
        if user_profiles is None:
            user_profiles = self.user_profiles
        dot_product = np.dot(user_profiles,
            self.infection_state*np.log(1-self.item_attributes))
        # Probability of being infected at the current iteration
        predicted_scores = 1 - np.exp(dot_product)
        self.log('System updates predicted scores given by users (rows) ' + \
            'to items (columns):\n' + str(predicted_scores))
        return predicted_scores

    def run(self, timesteps=50, startup=False, train_between_steps=True,
            repeated_items=True):
        """ Overrides run method of parent class :class:`Recommender`, so that repeated_items
            defaults to True in SIR models.

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
        Recommender.run(self, timesteps=timesteps, startup=startup,
                        train_between_steps=train_between_steps,
                        repeated_items=repeated_items)


    def draw_diffusion_tree(self):
        self.measurements[0].draw_tree()

    def get_structural_virality(self):
        return self.measurements[0].get_structural_virality()
