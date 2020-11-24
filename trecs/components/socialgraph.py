""" A binary social graph that represents relationships between users which
    can be used in different types of recommender systems
"""
import numpy as np
from trecs.logging import VerboseMode


class BinarySocialGraph(VerboseMode):
    """
    A mixin for classes with a
    :attr:`~trecs.models.recommender.BaseRecommender.users_hat` attribute
    to gain the basic functionality of a binary social graph.

    It assumes a network adjacency matrix of size `|U|x|U|`.
    """

    # expect these to be initialized
    users_hat = np.array([])
    num_users = np.array([])

    def follow(self, user_index, following_index):
        """
        Method to follow another user -- that is, to create a unidirectional
        link from one user to the other.

        Parameters
        ----------

        user_index: int
            Index of the user initiating the follow.

        following_index: int
            Index of the user to be followed.

        Raises
        ------

        ValueError
            If either of the user indices does not exist.
        """
        if user_index >= self.num_users or following_index >= self.num_users:
            raise ValueError(
                f"Number of users is {self.num_users}, but indices "
                f"{user_index} and {following_index} were requested."
            )
        if self.users_hat[following_index, user_index] == 0:
            self.users_hat[following_index, user_index] = 1
        elif self.is_verbose():
            self.log(f"User {following_index} was already following user {user_index}")

    def unfollow(self, user_index, following_index):
        """
        Method to unfollow another user -- that is, to delete the unidirectional
        link that goes from one user to the other.

        Parameters
        ----------

        user_index: int
            Index of the user initiating the unfollow.

        following_index: int
            Index of the user to be unfollowed.

        Raises
        ------

        ValueError
            If either of the user indices does not exist.
        """
        if user_index >= self.num_users or following_index >= self.num_users:
            raise ValueError(
                f"Number of users is {self.num_users}, but indices "
                f"{user_index} and {following_index} were requested."
            )
        if self.users_hat[following_index, user_index] == 1:
            self.users_hat[following_index, user_index] = 0
        elif self.is_verbose():
            self.log(f"User {following_index} was not following user {user_index}")

    def add_friends(self, user1_index, user2_index):
        """
        Method to add a user as *friends* -- that is, to create a bidirectional
        link that connects the two users.

        Parameters
        ----------

        user1_index: int
            Index of one user to establish the connection.

        user2_index: int
            Index of the other user to establish the connection.

        Raises
        ------

        ValueError
            If either of the user indices does not exist.
        """
        if user1_index >= self.num_users or user2_index >= self.num_users:
            raise ValueError(
                f"Number of users is {self.num_users}, but indices "
                f"{user1_index} and {user2_index} were requested."
            )
        if self.users_hat[user1_index, user2_index] == 0:
            self.users_hat[user1_index, user2_index] = 1
        elif self.is_verbose():
            self.log(f"User {user2_index} was already following user {user1_index}")
        if self.users_hat[user2_index, user1_index] == 0:
            self.users_hat[user2_index, user1_index] = 1
        elif self.is_verbose():
            self.log(f"User {user1_index} was already following user {user2_index}")

    def remove_friends(self, user1_index, user2_index):
        """
        Method to remove a user from *friends* -- that is, to remove a
        bidirectional link that connects the two users.

        Parameters
        ----------

        user1_index: int
            Index of one user for which to remove the connection.

        user2_index: int
            Index of the other user for which to remove the connection.

        Raises
        ------

        ValueError
            If either of the user indices does not exist.
        """
        if user1_index >= self.num_users or user2_index >= self.num_users:
            raise ValueError(
                f"Number of users is {self.num_users}, but indices "
                f"{user1_index} and {user2_index} were requested."
            )
        if self.users_hat[user1_index, user2_index] == 1:
            self.users_hat[user1_index, user2_index] = 0
        elif self.is_verbose():
            self.log(f"User {user2_index} was not following user {user1_index}")
        if self.users_hat[user2_index, user1_index] == 1:
            self.users_hat[user2_index, user1_index] = 0
        elif self.is_verbose():
            self.log(f"User {user1_index} was not following user {user2_index}")
