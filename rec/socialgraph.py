import numpy as np

from .debug import VerboseMode

class BinarySocialGraph(VerboseMode):
    """A mixin for classes with an :attr:`~Recommender.user_profiles` attribute
    to gain the basic functionality of a binary social graph.
    """
    def follow(self, user_index, following_index):
        # TODO allow for multiple indices
        if (user_index >= self.num_users or following_index >= self.num_users):
            raise ValueError("Number of user is %d, but indices %d and %d" + \
                            " were requested" % (self.num_users, user_index, following_index))
        if (self.user_profiles[following_index, user_index] == 0):
            self.user_profiles[following_index, user_index] = 1
        else:
            self.log("User %d was already following user %d" % (following_index,
                                                                user_index))

    def unfollow(self, user_index, following_index):
        # TODO allow for multiple indices
        if (user_index >= self.num_users or following_index >= self.num_users):
            raise ValueError("Number of user is %d, but indices %d and %d" + \
                            " were requested" % (self.num_users, user_index, following_index))
        if (self.user_profiles[following_index, user_index] == 1):
            self.user_profiles[following_index, user_index] = 0
        else:
            self.log("User %d was not following user %d" % (following_index, user_index))

    def add_friends(self, user1_index, user2_index):
        # TODO allow for multiple indices
        if (user1_index >= self.num_users or user2_index >= self.num_users):
            raise ValueError("Number of user is %d, but indices %d and %d" + \
                            " were requested" % (self.num_users, user1_index, user2_index))
        if (self.user_profiles[user1_index, user2_index] == 0):
            self.user_profiles[user1_index, user2_index] = 1
        else:
            self.log("User %d was already following user %d" % (user2_index, user1_index))
        if (self.user_profiles[user2_index, user1_index] == 0):
            self.user_profiles[user2_index, user1_index] = 1
        else:
            self.log("User %d was already following user %d" % (user1_index, user2_index))

    def remove_friends(self, user1_index, user2_index):
        # TODO allow for multiple indices
        if (user1_index >= self.num_users or user2_index >= self.num_users):
            raise ValueError("Number of user is %d, but indices %d and %d" + \
                            " were requested" % (self.num_users, user1_index, user2_index))
        if (self.user_profiles[user1_index, user2_index] == 1):
            self.user_profiles[user1_index, user2_index] = 0
        else:
            self.log("User %d was not following user %d" % (user2_index, user1_index))
        if (self.user_profiles[user2_index, user1_index] == 1):
            self.user_profiles[user2_index, user1_index] = 0
        else:
            self.log("User %d was not following user %d" % (user1_index, user2_index))
