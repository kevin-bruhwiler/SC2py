import numpy as np

class Noise():
	def __init__(self):
		self.epsilon = 1
		self.explore = 500000.
		self.mu = 0.0
		self.theta = 0.05 #0.60
		self.sigma = 0.90 #0.30

	def getNoise(self, x):
		self.epsilon -= 1.0 / self.explore
		n = []
		for a in x:
			n.append(max(self.epsilon, 0) * (self.theta * (self.mu - a) + self.sigma * np.random.random(1)))
		return np.asarray(n)
