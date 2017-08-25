import torch
import torch.nn as nn

class Actor(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(Actor, self).__init__()
		self.conv1 = nn.Conv2d(2, 4, 3, padding=1)
		self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
		self.conv3 = nn.Conv2d(8, 16, 3, padding=1)
		self.conv4 = nn.Conv2d(16, 32, 3, padding=1)
		self.maxpool = nn.MaxPool2d(2)
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, output_size)
		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()
		self.sigmoid = nn.Sigmoid()
		
	def forward(self, x):
		x = self.conv1(x)
		x = self.maxpool(x)
		x = self.relu(x)
		x = self.conv2(x)
		x = self.maxpool(x)
		x = self.relu(x)
		x = self.conv3(x)
		x = self.maxpool(x)
		x = self.relu(x)
		x = self.conv4(x)
		x = self.maxpool(x)
		x = x.view(x.size(0),-1)
		x = self.relu(x)
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		x = self.sigmoid(x)
		return x
		
class Critic(nn.Module):
	def __init__(self, input_size, hidden_size, num_actions):
		super(Critic, self).__init__()
		self.conv1 = nn.Conv2d(2, 4, 3, padding=1)
		self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
		self.conv3 = nn.Conv2d(8, 16, 3, padding=1)
		self.conv4 = nn.Conv2d(16, 32, 3, padding=1)
		self.maxpool = nn.MaxPool2d(2)
		self.fc1 = nn.Linear(input_size+num_actions, hidden_size)
		self.fc2 = nn.Linear(hidden_size, 1)
		self.relu = nn.ReLU()
		
	def forward(self, x, a):
		x = self.conv1(x)
		x = self.maxpool(x)
		x = self.relu(x)
		x = self.conv2(x)
		x = self.maxpool(x)
		x = self.relu(x)
		x = self.conv3(x)
		x = self.maxpool(x)
		x = self.relu(x)
		x = self.conv4(x)
		x = self.maxpool(x)
		x = x.view(x.size(0),-1)
		x = self.relu(x)
		a = a.view(a.size(0),-1)
		x = self.fc1(torch.cat([x,a],1))
		x = self.relu(x)
		x = self.fc2(x)
		return x
