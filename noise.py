import numpy as np

class Noise():
	def __init__(self):
		self.epsilon = 1
		self.explore = 50000.
		self.mu = 0.0
		self.theta = 0.60
		self.sigma = 0.30

	def getNoise(self, x):
		self.epsilon -= 1.0 / self.explore
return max(self.epsilon, 0) * (self.theta * (self.mu - x) + self.sigma * np.random.randn(1))
