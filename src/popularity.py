import numpy as np
import random
from abc import ABCMeta, abstractmethod
from recommender import Recommender
from measurements import Measurements
import matplotlib.pyplot as plt
from constants import *

plt.style.use('seaborn-whitegrid')

class PopularityRecommender(Recommender):
	def __init__(self, num_users, num_items):
		self.measurements = Measurements(num_items)
		self.theta_t = np.ones((num_users, 1))
		self.beta_t = np.zeros(num_items)

	# Stores interaction without training
	def store_interaction(self, interactions):
		self.beta_t = np.add(self.beta_t, interactions)

	# Trains model; it either adds new interactions,
	# or it updates the score with the stored interactions
	def train(self, interactions=None):
		if interactions is not None:
			self.beta_t = np.add(self.beta_t, interactions)
		self.s_t = self.beta_t

	# Return matrix that can be stored or used to train model
	def interact(self, preference):
		interactions = np.zeros(NUM_ITEMS)
		np.add.at(interactions, preference, 1)
		return interactions

	def measure_equilibrium(self, interactions):
		return self.measurements.measure_equilibrium(interactions)

	def generate_startup_interactions(self, num_startup_iter, num_items_per_iter=10, random_preference=True, preference=None):
		# Current assumptions:
		# 1. First (num_startup_iter * num_items_per_iter) items presented for startup
		# 2. New  num_items_per_iter items at each interaction, no recommendations
		# 3. TODO: consider user preferences that are not random
		if random_preference is True:
			preference = np.zeros(NUM_USERS * num_startup_iter)
		index = 0
		for t in range(1, num_startup_iter + 1):
			if random_preference is True:
				preference[index:index+NUM_USERS] = np.random.randint((t-1) * num_items_per_iter, t * num_items_per_iter, size=(NUM_USERS))
			index += NUM_USERS
		return self.interact(preference.astype(int))

	#def measure_equilibrium(self, interactions, interactions_old, i):
	#	self.delta_t[i] = np.trapz(interactions_old, dx=1) - np.trapz(interactions, dx=1)

	#def generate_interactions(self, num_iter, num_items_per_iter=10, num_new_items=5, random_preference=True, preference=None):
	def generate_interactions(self, num_items_per_iter=10, num_new_items=5, random_preference=True, preference=None):
		# Current assumptions:
		# 1. Interleave new items and recommended items, where recommended items appear at the beginning of the array
		# 2. Fixed number of new/recommended items
		# 3. All users are presented with the same items
		# 4. Each user interacts with different elements depending on preference
		# 5. Train after every interaction
		# TODO: each user can interact with each element at most once
		# TODO: consider user preferences that are not random
		num_recommended = num_items_per_iter - num_new_items
		#interacted = np.full((NUM_USERS, NUM_ITEMS), False)
		#user_row = np.arange(0, NUM_USERS)
		#i = 0
		#measure_interactions = np.zeros(NUM_ITEMS)
		#for t in range(num_iter * num_items_per_iter, NUM_ITEMS, num_new_items):
		# Assume 10 items per iteration (new *and* recommended)
		items = np.concatenate((self.recommend(k=num_recommended), np.arange(t, t + num_new_items))).astype(int)
		if random_preference is True:
			preference = np.random.randint(0, num_items_per_iter, size=(NUM_USERS)).astype(int)
		#interactions_old = np.copy(measure_interactions)
		#measure_interactions = self.interact(items[preference])
		interactions = items[preference]
		# sort interaction
		#measure_interactions[::-1].sort()
		#assert(np.sum(measure_interactions) == NUM_USERS)
		#self.measure_equilibrium(measure_interactions, interactions_old, i)
		#self.store_interaction(self.interact(interactions))
		#check = interacted[user_row, interactions] == False
		#interacted[user_row, interactions] = True
		#self.train()
		#i = (i + 1)
		return self.interact(interactions)
		#if np.all(check):
		#	continue
		# TODO: From here on, some user(s) has already interacted with the assigned item

	def recommend(self, k=1):
		# TODO: what if I consistently only do k=1? In that case I might want to think of just sorting once
		return self.s_t.argsort()[-k:][::-1]

if __name__ == '__main__':
	#plot_dim = int((NUM_ITEMS - (NUM_STARTUP_ITER * NUM_ITEMS_PER_ITER))/(int(NUM_ITEMS_PER_ITER/2)))
	rec = PopularityRecommender(NUM_USERS, NUM_ITEMS)
	# Startup
	startup_int = rec.generate_startup_interactions(NUM_STARTUP_ITER, num_items_per_iter=NUM_ITEMS_PER_ITER, random_preference=True)
	rec.train(startup_int)

	# Runtime
	for t in range(NUM_STARTUP_ITER * NUM_ITEMS_PER_ITER, NUM_ITEMS, int(NUM_ITEMS_PER_ITER / 2)):
		interactions = rec.generate_interactions(NUM_ITEMS_PER_ITER, int(NUM_ITEMS_PER_ITER / 2), True)
		rec.measure_equilibrium(interactions)
		rec.train(interactions)
	plt.plot(np.arange(len(rec.measurements.delta_t)), rec.measurements.delta_t)
	plt.show()
