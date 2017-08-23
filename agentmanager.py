import noise
import AC
import memory

from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch
_CUDA = torch.cuda.is_available()

from pysc2.lib import actions
from pysc2.lib import features
from pysc2.agents import base_agent

import numpy as np
import agentmanager
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]

class Agent(base_agent.BaseAgent):	
	def __init__(self):
		super(Agent, self).__init__()
		self.noise = noise.Noise()
		self.actor = AC.Actor(256, 256, 2)
		self.critic = AC.Critic(256, 256, 2)
		self.memory = memory.Memory(1000000)
		self.batch_size = 32
		self.loss = nn.MSELoss()
		self.actor_optim = optim.Adam(self.actor.parameters(), lr=0.0001)
		self.critic_optim = optim.Adam(self.critic.parameters(), lr=0.0001)
		if _CUDA:
			self.actor.cuda()
			self.critic.cuda()
	
	def setup(self, obs_spec, action_spec):
		super(Agent, self).setup(obs_spec, action_spec)

	def toVariable(self, x):
		if _CUDA:
			return Variable(torch.from_numpy(x).float()).cuda()
		else:
			return Variable(torch.from_numpy(x).float())

	def predictReward(self, x, a):
		return self.critic.forward(x, a)
		
	def getReward(self, x, a):
		y = self.critic.forward(x, a)
		return y.cpu().data.numpy().astype(float)
		
	def predictAction(self, x):
		return self.actor.forward(x)

	def step(self, obs, x, noise=True):
		super(Agent, self).step(obs)
		if _MOVE_SCREEN in obs.observation['available_actions']:
			y = self.actor.forward(self.toVariable(x))
			action = y.cpu().data.numpy().astype(float)
			if noise:
				action += self.noise.getNoise(action)
			action = (np.clip(action, 0, 63)).tolist()[0]
			return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, action]), action
		else:
			return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL]), [-1,-1]
		
	def train(self, observation, action, rewards, new_observation):
		self.memory.add((observation, action, rewards, new_observation))
		obs, acts, rewards, new_obs = self.memory.sample(self.batch_size)
		
		reward_labels = []
		for i in range(len(new_obs)):
			if new_obs[i].any():
				n_o = self.toVariable(np.asarray(new_obs[i])).unsqueeze(0)
				future_actions = self.predictAction(n_o)
				future_rewards = self.getReward(n_o, future_actions)
				reward_labels.append(rewards[i] + 0.9*future_rewards[0])
			else:
				reward_labels.append(rewards[i])
		reward_labels = np.asarray(reward_labels).astype(float)
				
		predicted_rewards = self.predictReward(self.toVariable(obs), self.toVariable(acts))
		reward_loss = self.loss(predicted_rewards, self.toVariable(reward_labels))
		self.critic_optim.zero_grad()
		reward_loss.backward()
		self.critic_optim.step()
				
		acts = self.predictAction(self.toVariable(obs))
		rewards = self.predictReward(self.toVariable(obs), acts)
		rewards = -torch.mean(rewards)
		self.actor_optim.zero_grad()
		rewards.backward()
		self.actor_optim.step()
