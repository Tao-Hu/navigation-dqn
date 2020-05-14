import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
LR = 5e-4               # learning rate
UPDATE_EVERY = 1000     # how often to update the target network
ALPHA = 0.6             # alpha parameter for prioritized experience replay
EPS = 1e-6              # small constant for prioritized experience replay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, state_size, action_size, seed, 
                 hidden_layers, drop_p = 0.5, 
                 double_dqn = False, prioritized_replay = False):
        """Initialize an Agent object.

           Arguments
           ---------
           state_size (int): Dimension of state
           action_size (int): Total number of possible actions
           seed (int): Random seed
           hidden_layers (list): List of integers, each element represents for the size of a hidden layer
           drop_p (float): Dropout probability
           double_dqn (logic): Indicator of using double DQN, True or False
           prioritized_replay (logic): Indicator of using prioritized experience replay, True or False
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.double_dqn = double_dqn
        self.prioritized_replay = prioritized_replay

        # Q network
        self.qnetwork_local = QNetwork(state_size, action_size, seed, hidden_layers, drop_p).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, hidden_layers, drop_p).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr = LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        # time step
        self.t_step = 0

    def act(self, state, eps = 0.0):
        """Epsilon-greedy policy based on current estimated Q values."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def step(self, state, action, reward, next_state, done):
        """One step during training."""
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Update policy Q network weights
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

        # Update target Q network weights every UPDATE_EVERY steps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
                target_param.data.copy_(local_param.data)

    def learn(self, experiences, gamma):
        """Update policy Q netowrk weights using given batch of experience tuples."""
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values of next states from the target Q network
        if not self.double_dqn:
            # DQN with fixed Q target
            Q_target_next = self.qnetwork_target(next_states).detach().max(1, keepdim = True)[0]
        else:
            # Double DQN
            max_actions = self.qnetwork_local(next_states).detach().max(1, keepdim = True)[1]
            Q_target_next = self.qnetwork_target(next_states).gather(1, max_actions)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_target_next * (1 - dones))

        # Get current estimated Q values from policy Q network
        Q_estimated = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_estimated, Q_targets)
        # Minimize the loss and update weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class ReplayBuffer():
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

           Arguments
           ---------
           action_size (int): Total number of possible actions
           buffer_size (int): Maximum size of buffer
           batch_size (int): Size of each training batch
           seed (int): Random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen = buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names = ["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_action, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_action, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k = self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None])).astype(np.uint8).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class PrioritizedReplayBuffer():
    def __init__(self, action_size, buffer_size, batch_size, seed, alpha = ALPHA, eps = EPS):
        """Initialize a Prioritized ReplayBuffer object.

           Arguments
           ---------
           action_size (int): Total number of possible actions
           buffer_size (int): Maximum size of buffer
           batch_size (int): Size of each training batch
           seed (int): Random seed
           alpha (float): Alpha parameter
           eps (float): small constant added to priority
        """
        self.action_size = action_size
        self.memory = deque(maxlen = buffer_size)
        self.priority = deque(maxlen = buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names = ["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.alpha = alpha
        self.eps = eps
        self.max_priority = 1.0