import numpy as np

"""Wrapper around numpy.random.Generator
"""
class Generator(np.random.Generator):
    def __init__(self, bitgenerator=None):
        if bitgenerator is None:
            bitgenerator = np.random.PCG64()
        np.random.Generator.__init__(self, bitgenerator)
