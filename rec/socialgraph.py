import numpy as np
import networkx as nx

from .debug import VerboseMode

class SocialGraph():
    @staticmethod
    def generate_random_graph(*args, **kwargs):
        """Thin wrapper around networkx's random graph generators"""
        graph_type = kwargs.pop('graph_type', None)
        if graph_type is None:
            default = kwargs.pop('default', None)
            graph_type = default if default is not None else nx.fast_gnp_random_graph
        graph = graph_type(*args, **kwargs)
        return nx.convert_matrix.to_numpy_array(graph)

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
