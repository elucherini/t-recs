import numpy as np


class Generator(np.random.Generator):
    """Wrapper around numpy.random.Generator"""
    def __init__(self, bitgenerator=None):
        if bitgenerator is None:
            bitgenerator = np.random.PCG64()
        np.random.Generator.__init__(self, bitgenerator)


class SocialGraphGenerator():
    @staticmethod
    def generate_random_graph(*args, **kwargs):
        """Thin wrapper around networkx's random graph generators"""
        import networkx as nx
        graph_type = kwargs.pop('graph_type', None)
        if graph_type is None:
            default = kwargs.pop('default', None)
            graph_type = default if default is not None else nx.fast_gnp_random_graph
        graph = graph_type(*args, **kwargs)
        return nx.convert_matrix.to_numpy_array(graph)
