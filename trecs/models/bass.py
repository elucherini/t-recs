"""
Bass Model for modeling the spread of infection. This can be applied to studying
virality in online communications.
"""
import networkx as nx
import numpy as np
from trecs.components import BinarySocialGraph
from trecs.components import Component
from trecs.random import Generator, SocialGraphGenerator
from trecs.metrics import StructuralVirality
from trecs.utils import (
    get_first_valid,
    is_array_valid_or_none,
    all_besides_none_equal,
    all_none,
)
from .recommender import BaseRecommender


class InfectionState(Component):  # pylint: disable=too-many-ancestors
    """Component that tracks infection state, which is a binary array with
    an element recording whether each user is infected
    """

    def __init__(self, infection_state=None, verbose=False):
        self.name = "infection_state"
        Component.__init__(
            self, current_state=infection_state, size=None, verbose=verbose, seed=None
        )


class InfectionThresholds(Component):  # pylint: disable=too-many-ancestors
    """Component that tracks infection thresholds, where each user has their own
    threshold for infection
    """

    def __init__(self, infection_thresholds=None, verbose=False):
        self.name = "infection_thresholds"
        Component.__init__(
            self,
            current_state=infection_thresholds,
            size=None,
            verbose=verbose,
            seed=None,
        )


class BassModel(BaseRecommender, BinarySocialGraph):
    """Bass model that, for now, only supports one item at a time"""

    def __init__(  # pylint: disable-all
        self,
        num_users=100,
        num_items=1,
        infection_state=None,
        infection_thresholds=None,
        item_representation=None,
        user_representation=None,
        actual_user_scores=None,
        actual_item_representation=None,
        probabilistic_recommendations=False,
        verbose=False,
        num_items_per_iter=1,
        seed=None,
        **kwargs
    ):
        # these are not allowed to be None at the same time
        if all_none(user_representation, num_users, infection_state):
            raise ValueError(
                "user_representation, num_users, and infection_state can't be all None"
            )
        if all_none(item_representation, num_items, infection_state):
            raise ValueError(
                "item_representation, num_items, and infection_state can't be all None"
            )
        if not is_array_valid_or_none(user_representation, ndim=2):
            raise ValueError("user_representation is invalid")
        if not is_array_valid_or_none(infection_state, ndim=2):
            raise TypeError("infection_state is invalid")
        if not is_array_valid_or_none(item_representation, ndim=2):
            raise ValueError("item_representation is invalid")
        if not is_array_valid_or_none(infection_thresholds, ndim=2):
            raise ValueError("infection_thresholds is invalid")

        # Determine num_users, give priority to user_representation
        # At the end of this, user_representation and num_users should not be None
        # In the arguments, I either get shape[0], or None if the matrix is None
        num_users = get_first_valid(
            getattr(user_representation, "shape", [None])[0],
            getattr(infection_state, "shape", [None])[0],
            num_users,
        )
        # Determine num_items, give priority to item_representation
        # At the end of this, item_representation should not be None
        num_items = get_first_valid(
            getattr(item_representation, "shape", [None, None])[1],
            getattr(infection_state, "shape", [None, None])[1],
            num_items,
        )
        generator = Generator(seed)
        if item_representation is None:
            item_representation = generator.uniform(size=(1, num_items))
        # if the actual item representation is not specified, we assume
        # that the recommender system's beliefs about the item attributes
        # are the same as the "true" item attributes
        if actual_item_representation is None:
            actual_item_representation = np.copy(item_representation)
        if user_representation is None:
            user_representation = SocialGraphGenerator.generate_random_graph(
                num=num_users, p=0.3, seed=seed, graph_type=nx.fast_gnp_random_graph
            )

        # Define infection_state
        if infection_state is None:
            infection_state = np.zeros((num_users, num_items))
            infected_users = generator.integers(num_users)
            infectious_items = generator.integers(num_items)
            infection_state[infected_users, infectious_items] = 1
        self.infection_state = infection_state

        if not all_besides_none_equal(
            getattr(user_representation, "shape", [None])[0],
            getattr(user_representation, "shape", [None, None])[1],
        ):
            raise ValueError("user_representation should be a square matrix")
        if not all_besides_none_equal(
            getattr(user_representation, "shape", [None])[0],
            getattr(infection_state, "shape", [None])[0],
        ):
            raise ValueError(
                "user_representation and infection_state should be of " + "same size on dimension 0"
            )
        if not all_besides_none_equal(
            getattr(item_representation, "shape", [None, None])[1],
            getattr(infection_state, "shape", [None, None])[1],
        ):
            raise ValueError(
                "item_representation and infection_state should be of " + "same size on dimension 1"
            )

        if infection_thresholds is None:
            infection_thresholds = abs(generator.uniform(size=(1, num_users)))

        self.infection_state = InfectionState(infection_state)
        self.infection_thresholds = InfectionThresholds(infection_thresholds)
        measurements = [StructuralVirality(np.copy(infection_state))]
        system_state = [self.infection_state]
        # Initialize recommender system
        # NOTE: Forcing to 1 item per iteration
        num_items_per_iter = 1
        BaseRecommender.__init__(
            self,
            user_representation,
            item_representation,
            actual_user_scores,
            actual_item_representation,
            num_users,
            num_items,
            num_items_per_iter,
            probabilistic_recommendations=False,
            measurements=measurements,
            system_state=system_state,
            verbose=verbose,
            seed=seed,
            score_fn=self.infection_probabilities,
            **kwargs
        )

    def initialize_user_scores(self):
        """
        If the Users object does not already have known user-item scores,
        then we calculate these scores.
        """
        # users compute their own scores using the true item attributes,
        # unless their own scores are already known to them
        self.users.set_score_function(self.infection_probabilities)
        if self.users.get_actual_user_scores() is None:
            self.users.compute_user_scores(self.items)

    def _update_user_profiles(self, interactions):
        """Private function that updates user profiles with data from
            latest interactions.

            Specifically, this function converts interactions into item attributes.
            For example, if user :math:`u` has interacted with item :math:`i`,
            then item :math:`i`'s attributes will be updated to increase the
            similarity between :math:`u` and :math:`i`.

        Args:
            interactions (numpy.ndarray): An array of item indices that users have
                interacted with in the latest step. Namely, interactions_u represents
                the index of the item that the user has interacted with.

        """
        infection_probabilities = self.predicted_scores[self.users.user_vector, interactions]
        newly_infected = np.where(infection_probabilities > self.infection_thresholds)
        if newly_infected[0].shape[0] > 0:
            self.infection_state[newly_infected[1], interactions[newly_infected[1]]] = 1

    def infection_probabilities(self, user_profiles, item_attributes):
        """Calculates the infection probabilities for each user at the current
        timestep
        Args:

        user_profiles: :obj:`array_like`
            First factor of the dot product, which should provide a
            representation of users.

        item_attributes: :obj:`array_like`
            Second factor of the dot product, which should provide a
            representation of items.
        """
        # This formula comes from Goel et al., The Structural Virality of Online Diffusion
        dot_product = np.dot(user_profiles, self.infection_state * np.log(1 - item_attributes))
        # Probability of being infected at the current iteration
        predicted_scores = 1 - np.exp(dot_product)
        return predicted_scores

    def run(self, timesteps=50, startup=False, train_between_steps=True, repeated_items=True):
        """Overrides run method of parent class :class:`Recommender`, so that
        repeated_items defaults to True in Bass models.

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
        # NOTE: force repeated_items to True
        repeated_items = True
        BaseRecommender.run(
            self,
            timesteps=timesteps,
            startup=startup,
            train_between_steps=train_between_steps,
            repeated_items=repeated_items,
        )

    def draw_diffusion_tree(self):
        """ Draw diffusion tree using matplotlib """
        for metric in self.metrics:
            if hasattr(metric, "draw_tree"):
                metric.draw_tree()

    def get_structural_virality(self):
        """ Return the value of the structural virality metric """
        for metric in self.metrics:
            if hasattr(metric, "get_structural_virality"):
                return metric.get_structural_virality()
        raise ValueError("Structural virality metric undefined")
