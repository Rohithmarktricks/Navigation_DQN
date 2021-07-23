"""

DQN agent source code
@author: Rohith Banka
Navigation Project done as a part of Deep Reinforcement Learning Nanodegree.
Initial code has been developed by the team of Udacity DRL Team, 2021.


The main source code for the Agent, to interact with the environment.
The input or the state is a vector.

Actions
=======
The simulation contains a single agent that navigates a large environment.
At each time step, it has four actions at its disposal:

0 - walk forward
1 - walk backward
2 - turn left
3 - turn right


states
======
The state space has 37 dimensions and contains the agent's velocity, 
along with ray-based perception of objects around agent's forward direction. 
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.


Learning
========
Agent learns every 4 time steps.
DQN and Double DQN has been implemented and can be selected by replaing the keyword argument "dqn_type" while initializing the agent. 
Network architecture would remain the same, however, the action selection is done by the main network in DDQN.

"""



# import the required libraries and modules.
import numpy as np
import random
from collections import namedtuple, deque
from model import QNetwork
import torch
import torch.nn.functional as F
import torch.optim as optim
from replay_buffer import ReplayBuffer, get_device



# Hyper-parameters, can be tuned to improve the performance.
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3
LR = 5e-4
UPDATE_EVERY = 4


# set the device the train the model i.e., CPU or GPU.
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = get_device()


class Agent():
    '''
    This is the agent/software/program that observes the environment, takes action, gets reward at each step.
    '''
    def __init__(self, state_size, action_size, dqn_type='DQN', replay_buffer_size=1e5, batch_size=64, gamma=0.99,
        learning_rate=1e-3, target_tau=2e-3, update_rate=4, seed=0):

        '''Parameters:
            state_size: dimension of the state (37)
            action_size: 4 possible actions
            dqn_type : DQN/DDQN 
            replay_buffer_size: size of the replay buffer
            batch_size: size of the memory batch for Learning and updating the weights of the target network.
            gamma: Discounting factor/ parameter to set the discounted value of the future rewards.
            learning_rate: Learing rate of the Neural Network.
            seed: This seed would be useful to have the same initialization everytime Agent is initialized.
        ''' 
        
        self.state_size = state_size
        self.action_size = action_size
        self.dqn_type = dqn_type
        self.buffer_size = int(replay_buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma

        self.learning_rate = learning_rate
        self.tau = target_tau
        self.update_rate = update_rate
        self.seed = int(seed)

        self.network = QNetwork(state_size, action_size, seed).to(device)
        self.target_network = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)

        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)

        self.t_step = 0


    def step(self, state, action, reward, next_state, done):
        '''This method saves the experience of the agent in the replay memory,
        and target network learns after every 4 steps.
        Input Parameters:

        state: Present state encountered by the agent in the environment.
        action: action that has been suggested by the network(DQN)
        reward: Discounted reward
        next_state: next_state(vector of 37 dimensions) that is the result of the action taken by the agent.
        done: boolean values to tell if the agent has reached the final state.

        '''
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_rate
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.get_batch()
                self.learn(experiences, self.gamma)


    def act(self, state, eps=0.0):
        '''Retuns actions for the given states as per current policy.'''
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.network.eval()
        with torch.no_grad():
            action_values = self.network(state)
        self.network.train()


        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()).astype(np.int32)
        else:
            return random.choice(np.arange(self.action_size)).astype(np.int32)


    def learn(self, experiences, gamma):
        '''experiences is a batch of namedtuple and is used to train the target network
        
        Input Parameters:
        experiences is a tuple of (states, actions, rewards, next_states, dones).
        This tuple of states, actions would be used to learn and update the weights matrices of the main and target network. 

        '''

        states, actions, rewards, next_states, dones = experiences

        qsa = self.network(states).gather(1, actions)

        if (self.dqn_type == 'DDQN'):
            '''Double DQN is a special case, where the action that yeilds max reward is taken from the network itself'''
            qsa_prime_actions = self.network(next_states).detach().max(1)[1].unsqueeze(1)
            qsa_prime_targets = self.target_network(next_states)[qsa_prime_actions].unsqueeze(1)
        else:
            '''DQN doesn't select specific actions, so the target network action predictions are used'''
            qsa_prime_target_values = self.target_network(next_states).detach()
            qsa_prime_targets = qsa_prime_target_values.max(1)[0].unsqueeze(1)

        qsa_targets = rewards + (gamma * qsa_prime_targets * (1 - dones))

        # Mean square loss between main network and target network predictions
        loss = F.mse_loss(qsa, qsa_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # To update the weights of the target network, using the weights of the main network.
        self.soft_update(self.network, self.target_network, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Rule of Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
