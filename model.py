"""
Neural network model for Vector based observation DQN agent.
This model has 3 hidden layers with Relu activation, PyTorch framework has been used to develop it.


@author : Rohith Banka

This code is taken from the Inital code that has been developed by Udactiy DRL Nanodegree Team, 2021.

Further References: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

"""

import torch
import torch.nn as nn
import torch.nn.functional as F



class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed, layer1=128, layer2=64, layer3=32):
        super().__init__()
        self.seed = seed
        self.fc1 = nn.Linear(state_size, layer1)
        self.fc2 = nn.Linear(layer1, layer2)
        self.fc3 = nn.Linear(layer2, layer3)
        self.fc4 = nn.Linear(layer3, action_size)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)