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
import math, copy
import agentmanager
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]
_NO_OP = actions.FUNCTIONS.no_op.id

class Agent(base_agent.BaseAgent):	
	def __init__(self):
		super(Agent, self).__init__()
		self.tau = 0.001
		self.noise = noise.Noise()
		self.actor = AC.Actor(4096, 4096, 1)
		self.critic = AC.Critic(4096, 4096, 1)
		self.target_critic = copy.deepcopy(self.critic)
		self.memory = memory.Memory(1000000)
		self.batch_size = 32
		self.loss = nn.MSELoss()
		self.actor_optim = optim.Adam(self.actor.parameters(), lr=0.0001)
		self.critic_optim = optim.Adam(self.critic.parameters(), lr=0.0001)
		if _CUDA:
			self.actor.cuda()
			self.critic.cuda()
			self.target_critic.cuda()
		self.steps = 0
		self.current_action = None
		self.current_location = [-1,-1]
	
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
		y = self.target_critic.forward(x, a)
		return y.cpu().data.numpy().astype(float)
		
	def predictAction(self, x):
		return self.actor.forward(x)
		
	def updateTargetNet(self):
		c = self.critic.state_dict()
		t = self.target_critic.state_dict()
		new_t = {}
		for k in c:
			new_t[k] = (c[k]*self.tau) + (t[k]*(1.-self.tau))
		self.target_critic.load_state_dict(new_t)
		return
		
	def reset(self):
		self.steps = 0
		return
	
	def getLocation(self, selected):
		indexes = np.where(selected == 1)
		count, x, y = (0,0,0)
		for i in zip(indexes[2], indexes[3]):
			count += 1
			x += i[0]
			y += i[1]
		if count == 0:
			return
		x /= count
		y /= count
		self.current_location = [x , y]
		return 
		
	def calculateTarget(self, action, selected):
		x , y = self.current_location
		theta = math.radians(action * 360)
		w = selected.shape[3]-1
		h = selected.shape[2]-1
		radius = 3
		X = np.clip(radius * math.cos(theta) + x,0,w)
		Y = np.clip(radius * math.sin(theta) + y,0,h)
		return [X, Y]
		
	def step(self, obs, x, selected, noise=True):
		super(Agent, self).step(obs)
		self.steps += 1
		if _MOVE_SCREEN in obs.observation['available_actions']:
			self.getLocation(selected)
			#if not self.steps % 1 == 0:
				#return actions.FunctionCall(_NO_OP, []), self.current_action
			y = self.actor.forward(self.toVariable(x))
			action = y.cpu().data.numpy().astype(float)
			if noise:
				action += self.noise.getNoise(action)
				action = np.clip(action, 0, 1)
			self.current_action = action
			target = self.calculateTarget(action[0][0], selected)
			return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target]), action
		else:
			return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL]), None
		
	def train(self, observation, action, rewards, new_observation):
		self.memory.add((observation, action, rewards, new_observation))
		obs, acts, rewards, new_obs = self.memory.sample(self.batch_size)
		
		reward_labels = []
		for i in range(len(new_obs)):
			if not new_obs[i] is None:
				n_o = self.toVariable(np.asarray(new_obs[i]))
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
		self.updateTargetNet()
				
		acts = self.predictAction(self.toVariable(obs))
		rewards = self.predictReward(self.toVariable(obs), acts)
		rewards = -torch.mean(rewards)
		self.actor_optim.zero_grad()
		rewards.backward()
		self.actor_optim.step()
