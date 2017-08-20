from collections import deque
import numpy as np
import random

class Memory():
	def __init__(self, max_size):
		self.mem = deque([], max_size)
		
	def add(self, memory):
		self.mem.append(memory)
		
	def sample(self, num):
		if num > len(self.mem):
			batch = list(self.mem)
		else:
			batch = random.sample(self.mem, num)
		observations = np.asarray([b[0] for b in batch])
		actions = np.asarray([b[1] for b in batch])
		rewards = np.asarray([b[2] for b in batch])
		new_observations = [b[3] for b in batch]
		return observations, actions, rewards, new_observations
