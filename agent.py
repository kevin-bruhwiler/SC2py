from pysc2.agents import base_agent
from pysc2.lib import features
from pysc2.lib import actions

import noise
import AC
import memory

import numpy as np

from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch
_CUDA = torch.cuda.is_available()

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
		self.noise = noise.Noise()
		self.actor = AC.Actor(400, 400, 2)
		self.critic = AC.Critic(400, 400, 2)
		self.memory = memory.Memory(1000000)
		self.batch_size = 32
		self.loss = nn.MSELoss()
		self.actor_optim = optim.Adam(self.actor.parameters(), lr=0.0001)
		self.critic_optim = optim.Adam(self.critic.parameters(), lr=0.0001)
		self.observation = None
		self.first = True
		if _CUDA:
			self.actor.cuda()
			self.critic.cuda()
	
	def toVariable(self, x):
		if _CUDA:
			return Variable(torch.from_numpy(x).float()).cuda()
		else:
			return Variable(torch.from_numpy(x).float())

	def predictReward(self, x, a):
		return self.critic.forward(x, a)
		
	def getReward(self, x, a):
		y = self.critic.forward(x, a)
		if self.cuda:
			return y.cpu().data.numpy().astype(float)
		return y.data.numpy().astype(float)
		
	def predictAction(self, x):
		return self.actor.forward(x)

	def getAction(self, x, noise=False):
		y = self.actor.forward(self.toVariable(x))
		action = y.cpu().data.numpy().astype(float)
		if noise:
			return self.noise.getNoise(action)+action
		else:
			return action
	
	def step(self, obs):
		super(Agent, self).step(obs)
		if _MOVE_SCREEN in obs.observation['available_actions']:
			new_observation = obs.observation["screen"][_PLAYER_RELATIVE].reshape((1,84,84))
			action = self.getAction(new_observation.reshape((1,1,84,84)), noise=True)
			target = np.clip(action*84, 0, 83)[0].tolist()
			reward = sum(obs.observation['score_cumulative'])
			if self.first:
				self.first = False
				self.observation = np.zeros_like(new_observation)
			self.memory.add((self.observation, action, reward, new_observation))
			obs, acts, rewards, new_obs = self.memory.sample(self.batch_size)

			reward_labels = []
			for i in range(len(new_obs)):
				if not new_obs[i].any(None):
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
		
			self.observation = new_observation
			return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])
		else:
			return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
		return
