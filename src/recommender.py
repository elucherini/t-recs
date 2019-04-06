import numpy as np
from abc import ABCMeta, abstractmethod

# Recommender systems: abstract class
class Recommender(metaclass=ABCMeta):
	def __init__(self, theta_t, beta_t, delta_t = None):
		self.s_t = None
		self.theta_t = theta_t
		self.beta_t = beta_t

		# Measurements
		self.delta_t = delta_t

	def train(self):
		self.s_t = np.dot(self.theta_t, self.beta_t)

	@abstractmethod
	def interact(self):
		pass

	@abstractmethod
	def measure_equilibrium(self):
		pass

	@abstractmethod
	def generate_interactions(self):
		pass

	@abstractmethod
	def generate_startup_interactions(self):
		pass
