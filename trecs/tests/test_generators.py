import numpy as np
from trecs.random import Generator, SocialGraphGenerator
import pytest


class TestGenerator:
    def test_default(self, seed=None):
        gen = Generator()

        assert isinstance(gen.bit_generator, np.random.PCG64)

        if seed is None:
            seed = np.random.randint(1, 1000)
        # check that we're setting the seed correctly in numpy
        gen1 = Generator(seed=seed)
        gen2 = Generator(seed=seed)
        assert isinstance(gen1.bit_generator, np.random.PCG64)
        assert isinstance(gen2.bit_generator, np.random.PCG64)
        assert gen1.normal() == gen2.normal()
        assert gen1.binomial(3, 0.5) == gen2.binomial(3, 0.5)

    def test_bit_generators(self, seed=None):
        # check that we're setting bit generators correctly
        mt = np.random.MT19937()
        pcg64 = np.random.PCG64()
        philox = np.random.Philox()
        sfc64 = np.random.SFC64()

        gen = Generator(bit_generator=mt)
        assert isinstance(gen.bit_generator, np.random.MT19937)
        gen = Generator(bit_generator=pcg64)
        assert isinstance(gen.bit_generator, np.random.PCG64)
        gen = Generator(bit_generator=philox)
        assert isinstance(gen.bit_generator, np.random.Philox)
        gen = Generator(bit_generator=sfc64)
        assert isinstance(gen.bit_generator, np.random.SFC64)

        # also test that setting both bit_generators and seed throws a warning
        if seed is None:
            seed = np.random.randint(10000)
        with pytest.warns(UserWarning):
            gen = Generator(seed=seed, bit_generator=mt)


class TestSocialGraphGenerator:
    def test_default(self, n=None, seed=None):
        if n is None:
            n = np.random.randint(1000)
        if seed is None:
            seed = np.random.randint(10000)
        graph = SocialGraphGenerator.generate_random_graph(num=n)

        assert isinstance(graph, np.ndarray)
        assert graph.shape == (n, n)

        # test seeds are set correctly
        graph1 = SocialGraphGenerator.generate_random_graph(num=n, seed=seed)
        graph2 = SocialGraphGenerator.generate_random_graph(num=n, seed=seed)
        assert np.array_equal(graph1, graph2)

    def test_graph_generators(self, n=None, seed=None):
        import networkx as nx

        if n is None:
            n = np.random.randint(1000)
        if seed is None:
            seed = np.random.randint(10000)

        # test a few non-default random graph generators
        gnm = nx.gnm_random_graph
        # make sure m will always be valid
        m = np.random.randint(n)
        graph = SocialGraphGenerator.generate_random_graph(graph_type=gnm, num=n, m=m)

        assert isinstance(graph, np.ndarray)
        assert graph.shape == (n, n)
        assert np.count_nonzero(graph) == m * 2

        ws = nx.watts_strogatz_graph
        k = m
        graph = SocialGraphGenerator.generate_random_graph(graph_type=ws, num=n, k=k, p=0.5)
        assert isinstance(graph, np.ndarray)
        assert graph.shape == (n, n)
