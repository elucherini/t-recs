from .recommender import Recommender
from .social import SocialFiltering
from .stats import Distribution
from .measurement import StructuralVirality
import numpy as np
import math

class BassModel(SocialFiltering, Recommender):
    '''SIR model that, for now, only supports one item at a time'''
    def __init__(self, num_users=100, num_items=1, infection_state=None,
        item_representation=None, user_representation=None, infection_threshold=None,
        actual_user_scores=None, verbose=False, num_items_per_iter=10, num_new_items=30):
        # Give precedence to user_representation, otherwise build empty one
        if user_representation is None:
            if num_users is None and infection_state is None:
                raise ValueError("num_users, infection_state, and user_representation can't be all None")
            if infection_state is not None:
                num_users = infection_state.shape[0]
            user_representation = np.diag(np.diag(np.ones((num_users, num_users),
                                                          dtype=int)))
        elif (user_representation.shape[0] != user_representation.shape[1]):
            raise ValueError("user_representation should be a square matrix but it's %s" % (
                            str(user_representation.shape)))
        elif (infection_state is not None and
            user_representation.shape[0] != infection_state.shape[0]):
            raise ValueError("user_representation and infection_state should have same size on dim 0" + \
                            "but they are sized %s and %s" % (str(user_representation.shape),
                                str(infection_state.shape)))
        else:
            num_users = user_representation.shape[0]
        assert(num_users is not None)
        assert(user_representation is not None)
        assert(num_users == user_representation.shape[0] == user_representation.shape[1])
        # Give precedence to item_representation, otherwise build random one
        if item_representation is not None:
            if (infection_state is not None
                and infection_state.shape[1] != item_representation.shape[1]):
                raise ValueError("item_representation and infection_state should have same size on dim 1" + \
                            "but they are sized %s and %s" % (str(item_representation.shape),
                                str(infection_state.shape)))
            num_items = item_representation.shape[1]
        elif infection_state is not None:
            num_items = infection_state.shape[1]
        elif num_items is None:
            raise ValueError("num_items, infection_state, and item_representation can't be all None")
        else:
            item_representation = Distribution('uniform', size=(1,num_items)).compute()

        assert(num_items is not None)
        assert(item_representation is not None)
        if infection_state is None:
        # TODO change parameters
            infection_state = np.zeros((num_users, num_items))
            infection_state[np.random.randint(num_users), :] = 1
        assert(infection_state is not None)
        self.infection_state = infection_state
        # TODO support separate threshold for each user
        if infection_threshold is None or infection_threshold >= 1:
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
        #print("Interactions:\n", infection_probabilities)
        newly_infected = np.where(infection_probabilities > self.infection_threshold)
        #print("Newly infected:\n", newly_infected)
        if newly_infected[0].shape[0] > 0:
            assert(newly_infected[0].shape == interactions[newly_infected].shape)
            # this might not be true since now some users don't get infected at all and self.indices
            # assumes that *all* users are infected once per iteration. Needs to be tested and
            # be updated accordingly.
            #assert(self.infection_state[newly_infected, interactions[newly_infected]].all() == 0)
            self.infection_state[newly_infected, interactions[newly_infected]] = 1
            #print("New infection state:\n", self.infection_state)


    def train(self, user_profiles=None, item_attributes=None, normalize=False):
        """ Overrides train method of parent class :class:`Recommender`.

            Args:
                normalize (bool, optional): set to True if the scores should be normalized,
            False otherwise.

            TODO: Rewrite so super().train() takes the arguments of the dot product (AKA turns out
                    normalizing is not the only thing I might want to do)
        """
        # normalizing the user profiles is meaningless here
        # This formula comes from Goel et al., The Structural Virality of Online Diffusion
        #print(np.where(np.log(1 - self.item_attributes) == np.nan))
        if user_profiles is None:
            user_profiles = self.user_profiles
        dot_product = np.dot(user_profiles,
            self.infection_state*np.log(1-self.item_attributes))
        #print("Dot product (1-e^(dot_product)):\n", dot_product)
        # Probability of being infected at the current iteration
        predicted_scores = 1 - np.exp(dot_product)
        #print('Predicted scores:\n', self.predicted_scores)
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
