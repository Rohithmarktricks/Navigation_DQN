# import the required libraries and modules.
import numpy as np
import random
from collections import namedtuple, deque
from model import QNetwork
import torch
import torch.nn.functional as F
import torch.optim as optim
# from replay_buffer import ReplayBuffer, get_device



# Hyper-parameters
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3
LR = 5e-4
UPDATE_EVERY = 4


# set the device the train the model i.e., CPU or GPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = get_device()


class Agent():
    '''
    This is the agent/software/program that observes the environment, takes action, gets reward at each step.
    '''
    def __init__(self, state_size, action_size, dqn_type='DQN', replay_buffer_size=1e5, batch_size=64, gamma=0.99,
        learning_rate=1e-3, target_tau=2e-3, update_rate=4, seed=0):
        
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
        and target network learns after every 4 steps.'''
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


    def learn(self, experiences, gamma, DQN=True):
        '''experiences is a batch of namedtuple and is used to train the target network'''

        states, actions, rewards, next_states, dones = experiences

        qsa = self.network(states).gather(1, actions)

        if (self.dqn_type == 'DDQN'):

            qsa_prime_actions = self.network(next_states).detach().max(1)[1].unsqueeze(1)
            qsa_prime_targets = self.target_network(next_states)[qsa_prime_actions].unsqueeze(1)
        else:
            # qsa_prime_target_values = self.target_network(next_states).detach()
            qsa_prime_targets = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)

        qsa_targets = rewards + (gamma * qsa_prime_targets * (1 - dones))

        loss = F.mse_loss(qsa, qsa_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.network, self.target_network, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:


    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.seed = random.seed(seed)


    def add(self, state, action, reward, next_state, done):
        exp = self.experience(state, action, reward, next_state, done)
        self.memory.append(exp)


    def get_batch(self):
        
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)