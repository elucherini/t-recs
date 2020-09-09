import numpy as np

from rec.utils import VerboseMode


class BinarySocialGraph(VerboseMode):
    """
     A mixin for classes with a :attr:`~rec.models.recommender.BaseRecommender.user_profiles` attribute
    to gain the basic functionality of a binary social graph.

    It assumes a network adjacency matrix of size `|U|x|U|`.
    """

    def follow(self, user_index, following_index):
        """
        Method to follow another user -- that is, to create a unidirectional link from one user to the other.

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
                "Number of users is %d, but indices %d and %d"
                + " were requested" % (self.num_users, user_index, following_index)
            )
        if self.user_profiles[following_index, user_index] == 0:
            self.user_profiles[following_index, user_index] = 1
        else:
            self.log(
                "User %d was already following user %d" % (following_index, user_index)
            )

    def unfollow(self, user_index, following_index):
        """
        Method to unfollow another user -- that is, to delete the unidirectional link that goes from one user to the other.

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
                "Number of user is %d, but indices %d and %d"
                + " were requested" % (self.num_users, user_index, following_index)
            )
        if self.user_profiles[following_index, user_index] == 1:
            self.user_profiles[following_index, user_index] = 0
        else:
            self.log(
                "User %d was not following user %d" % (following_index, user_index)
            )

    def add_friends(self, user1_index, user2_index):
        """
        Method to add a user as *friends* -- that is, to create a bidirectional link that connects the two users.

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
                "Number of user is %d, but indices %d and %d"
                + " were requested" % (self.num_users, user1_index, user2_index)
            )
        if self.user_profiles[user1_index, user2_index] == 0:
            self.user_profiles[user1_index, user2_index] = 1
        else:
            self.log(
                "User %d was already following user %d" % (user2_index, user1_index)
            )
        if self.user_profiles[user2_index, user1_index] == 0:
            self.user_profiles[user2_index, user1_index] = 1
        else:
            self.log(
                "User %d was already following user %d" % (user1_index, user2_index)
            )

    def remove_friends(self, user1_index, user2_index):
        """
        Method to remove a user from *friends* -- that is, to remove a bidirectional link that connects the two users.

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
                "Number of user is %d, but indices %d and %d"
                + " were requested" % (self.num_users, user1_index, user2_index)
            )
        if self.user_profiles[user1_index, user2_index] == 1:
            self.user_profiles[user1_index, user2_index] = 0
        else:
            self.log("User %d was not following user %d" % (user2_index, user1_index))
        if self.user_profiles[user2_index, user1_index] == 1:
            self.user_profiles[user2_index, user1_index] = 0
        else:
            self.log("User %d was not following user %d" % (user1_index, user2_index))
