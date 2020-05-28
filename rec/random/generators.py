import numpy as np
import warnings


class Generator(np.random.Generator):
    """Wrapper around numpy.random.Generator"""
    def __init__(self, seed=None, bit_generator=None):
        if bit_generator is not None and seed is not None:
            warnings.warn("Seed has not been set. Please set seed in bit generator")
        if bit_generator is None:
            bit_generator = np.random.PCG64(np.random.SeedSequence(seed))
        np.random.Generator.__init__(self, bit_generator)


class SocialGraphGenerator():
    @staticmethod
    def generate_random_graph(n, *args, **kwargs):
        """Thin wrapper around networkx's random graph generators"""
        import networkx as nx
        if not isinstance(n, int):
            raise ValueError("n must be an integer")
        graph_type = kwargs.pop('graph_type', None)
        if graph_type is None:
            graph_type = nx.fast_gnp_random_graph
            kwargs['p'] = 0.5
        graph = graph_type(n=n, *args, **kwargs)
        return nx.convert_matrix.to_numpy_array(graph)

