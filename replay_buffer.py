"""
The input to the DNN that's being used in vector based. We store experience tuple 
in this Replay Buffer as the agent is interacting with the environment and then 
sample a small batch of tuples from it in order to learn.

As, a result, the agent will be able to learn from individual tuple multiple times,
recall rare occurrences/interactions, and in general make better use of past experiences.


@author : Rohith Banka

Project for Udacity NanoDegree in Deep Reinforcement Learning (DRLND)

code expanded and adaptyed from code examples provided by Udacity DRL Team, 2021.

Other References: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

"""


# Import the required packages/modules

import torch
import numpy as np
import random
from collections import deque, namedtuple



def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


class ReplayBuffer:


    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.seed = random.seed(seed)
        self.device = get_device()


    def add(self, state, action, reward, next_state, done):
        exp = self.experience(state, action, reward, next_state, done)
        self.memory.append(exp)


    def get_batch(self):
        
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)