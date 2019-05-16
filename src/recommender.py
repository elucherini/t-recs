import numpy as np
from abc import ABCMeta, abstractmethod

# Recommender systems: abstract class
class Recommender(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, theta_t, beta_t):
        self.s_t = None
        self.theta_t = theta_t
        self.beta_t = beta_t

    def train(self):
        self.s_t = np.dot(self.theta_t, self.beta_t)

    @abstractmethod
    def recommend(self, k=1):
        pass

    @abstractmethod
    def interact(self):
        pass

    @abstractmethod
    def interact_startup(self):
        pass
    
    @abstractmethod
    def measure_equilibrium(self, interactions):
        pass