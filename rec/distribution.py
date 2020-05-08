import numpy as np

"""Wrapper around numpy.random.Generator
"""
class Generator(np.random.Generator):
    def __init__(self, *args, **kwargs):
        np.random.Generator.__init__(self, np.random.PCG64(), *args, **kwargs)
