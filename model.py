import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden_layers, drop_p = 0.5):
        """Initialize parameters and build a Q network.

           Arguments
           ---------
           state_size (int): Dimension of state
           action_size (int): Total number of possible actions
           seed (int): Random seed
           hidden_layers (list): List of integers, each element represents for the size of a hidden layer
           drop_p (float): Dropout probability
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], action_size)
        self.dropout = nn.Dropout(p = drop_p)

    def forward(self, state):
        """Forward pass through the network."""
        for linear in self.hidden_layers:
            state = F.relu(linear(state))
            state = self.dropout(state)
        state = self.output(state)
        return state