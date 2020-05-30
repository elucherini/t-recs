import numpy as np
import warnings


class Generator(np.random.Generator):
    """
    Wrapper around :class:`numpy.random.Generator`. Please see the `Numpy documentation`_ for more details.

    .. _Numpy documentation: https://numpy.org/doc/stable/reference/random/generator.html

    Parameters
    -----------
        seed: int or None (optional, default: None)

        bit_generator: :class:`numpy.random.BitGenerator` or None (optional, default: None)
            :class:`numpy.random.BitGenerator`. Please see Numpy's `BitGenerator documentation`_ for more details.

    .. _BitGenerator documentation: https://numpy.org/doc/stable/reference/random/bit_generators/index.html
    """
    def __init__(self, seed=None, bit_generator=None):
        if bit_generator is not None and seed is not None:
            warnings.warn("Seed has not been set. Please set seed in bit generator")
        if bit_generator is None:
            bit_generator = np.random.PCG64(np.random.SeedSequence(seed))
        np.random.Generator.__init__(self, bit_generator)


class SocialGraphGenerator():
    """
    Thin wrapper around the `Networkx random graph generators`_. We use this static class to generate random network adjacency matrices.

    By default, it generates a `binomial graph`_, but it can generate any other random graph included in the Networkx API. Please refer to the Networkx documentation.

    .. _Networkx random graph generators: https://networkx.github.io/documentation/stable/reference/generators.html#module-networkx.generators.random_graphs
    .. _binomial graph: https://networkx.github.io/documentation/stable/reference/generated/networkx.generators.random_graphs.fast_gnp_random_graph.html#networkx.generators.random_graphs.fast_gnp_random_graph
    """
    @staticmethod
    def generate_random_graph(n, *args, **kwargs):
        """

        **Note:** to change type of graph, please include the `graph_type` parameter.

        Parameters
        -----------

            n: int
                Number of nodes in the graph. This is equivalent to the number of users in the system.

        Returns
        --------

            Adjacency matrix: :obj:`numpy.ndarray`
                Size `|U|x|U|`.

        Raises
        -------
            ValueError
                If `n` is not an integer.

        Examples
        ----------

            A minimal use case:

                >>> from rec.random import SocialGraphGenerator
                >>> n = 1000 # <-- number of nodes (users)
                >>> graph = SocialGraphGenerator.generate_random_graph(n=n)
                >>> graph # <-- 1000x1000 binomial adjacency matrix

            Changing random graph generator (e.g., with the `random_regular_graph`_):

            .. _random_regular_graph: https://networkx.github.io/documentation/stable/reference/generated/networkx.generators.random_graphs.random_regular_graph.html#networkx.generators.random_graphs.random_regular_graph

                >>> from networkx.generators.random_graphs import random_regular_graph
                >>> rrg = random_regular_graph
                >>> d = 30 # <-- degree, also required by random_regular_graph
                >>> graph = SocialGraphGenerator.generate_random_graph(n=n, d=d, graph_type=rrg)
        """
        import networkx as nx
        if not isinstance(n, int):
            raise ValueError("n must be an integer")
        graph_type = kwargs.pop('graph_type', None)
        if graph_type is None:
            graph_type = nx.fast_gnp_random_graph
            kwargs['p'] = 0.5
        graph = graph_type(n=n, *args, **kwargs)
        return nx.convert_matrix.to_numpy_array(graph)

