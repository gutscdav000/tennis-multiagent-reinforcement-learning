import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperparameters import *

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

# class Actor(nn.Module):
#     """Actor (Policy) Model."""

#     def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128):
#         """Initialize parameters and build model.
#         Params
#         ======
#             state_size (int): Dimension of each state
#             action_size (int): Dimension of each action
#             seed (int): Random seed
#             fc1_units (int): Number of nodes in first hidden layer
#             fc2_units (int): Number of nodes in second hidden layer
#         """
#         super(Actor, self).__init__()
#         self.seed = torch.manual_seed(seed)
#         self.fc1 = nn.Linear(state_size, fc1_units)
#         self.bn1 = nn.BatchNorm1d(fc1_units)
#         self.fc2 = nn.Linear(fc1_units, fc2_units)
#         self.fc3 = nn.Linear(fc2_units, action_size)
#         self.reset_parameters()

#     def reset_parameters(self):
#         self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
#         self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
#         self.fc3.weight.data.uniform_(-3e-3, 3e-3)

#     def forward(self, state):
#         """Build an actor (policy) network that maps states -> actions."""
#         x = F.relu(self.bn1(self.fc1(state)))
#         x = F.relu(self.fc2(x))
#         return F.tanh(self.fc3(x))


# class Critic(nn.Module):
#     """Critic (Value) Model."""

#     def __init__(self, state_size, action_size, seed, fcs1_units=256, fc2_units=128):
#         """Initialize parameters and build model.
#         Params
#         ======
#             state_size (int): Dimension of each state
#             action_size (int): Dimension of each action
#             seed (int): Random seed
#             fcs1_units (int): Number of nodes in the first hidden layer
#             fc2_units (int): Number of nodes in the second hidden layer
#         """
#         super(Critic, self).__init__()
#         self.seed = torch.manual_seed(seed)
#         self.fcs1 = nn.Linear(state_size, fcs1_units)
#         self.bn1 = nn.BatchNorm1d(fcs1_units)
#         self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
#         self.fc3 = nn.Linear(fc2_units, 1)
#         self.reset_parameters()

#     def reset_parameters(self):
#         self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
#         self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
#         self.fc3.weight.data.uniform_(-3e-4, 3e-4)

#     def forward(self, state, action):
#         """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
#         xs = F.relu(self.bn1(self.fcs1(state)))
#         x = torch.cat((xs, action), dim=1)
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)
    
    


# class Actor(torch.nn.Module):
#     def __init__(self, state_size, action_size):
#         super(Actor, self).__init__()
#         self.state_size   = state_size
#         self.action_size = action_size

#         self.fc1 = torch.nn.Linear(state_size, 24)
#         self.bn1 = torch.nn.BatchNorm1d(24)
#         self.fc2 = torch.nn.Linear(24, 24)
#         self.bn2 = torch.nn.BatchNorm1d(24)
#         self.out = torch.nn.Linear(24, action_size)
#         self.reset_parameters()
        
#     def forward(self, states):
#         batch_size = states.size(0)
#         x = self.fc1(states)
#         x = F.relu(self.bn1(x))
#         x = F.relu(self.bn2(self.fc2(x)))
#         x = F.tanh(self.out(x))
#         return x
    
#     def reset_parameters(self):
#         self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
#         self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
#         self.out.weight.data.uniform_(-3e-3, 3e-3)

# class Critic(torch.nn.Module):
#     def __init__(self, state_size, action_size):
#         super(Critic, self).__init__()
#         self.state_size   = state_size
#         self.action_size = action_size

#         self.fc1 = torch.nn.Linear(state_size, 24)
# #         self.fc1 = torch.nn.Linear(1, 24)
#         self.bn1 = torch.nn.BatchNorm1d(24)
#         self.fc2 = torch.nn.Linear(24+action_size, 24)
#         self.bn2 = torch.nn.BatchNorm1d(24)
#         self.fc3 = torch.nn.Linear(24, 12)
#         self.bn3 = torch.nn.BatchNorm1d(12)
#         self.out = torch.nn.Linear(12, 1)
#         self.reset_parameters()
        
#     def forward(self, states, actions):
#         batch_size = states.size(0)
#         print(states.shape)
#         states = torch.unsqueeze(states,0)
#         print(states.shape)
#         xs = F.leaky_relu(self.bn1(self.fc1(states)))
#         x = torch.cat((xs, actions), dim=1) #add in actions to the network
#         x = F.leaky_relu(self.bn2(self.fc2(x)))
#         x = F.leaky_relu(self.bn3(self.fc3(x)))
#         x = self.out(x)
#         return x
    
#     def reset_parameters(self):
#         self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
#         self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
#         self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
#         self.out.weight.data.uniform_(-3e-3, 3e-3)

class ActorModel(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, input_size, output_size, seed, fc1_units=24, fc2_units=24):
        """Initialize parameters and build actor model.
        Params
        ======
            input_size (int):  number of dimensions for input layer
            output_size (int): number of dimensions for output layer
            seed (int): random seed
            fc1_units (int): number of nodes in first hidden layer
            fc2_units (int): number of nodes in second hidden layer
        """
        super(ActorModel, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, output_size)
        self.bn = nn.BatchNorm1d(fc1_units)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights with near zero values."""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor network that maps states to actions."""

#         print(state.shape)
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)
#         print(state.shape)
#         print(self.fc1)
        x = self.fc1(state)
        x = F.relu(x)
        x = self.bn(x)
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x


class CriticModel(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, input_size, seed, fc1_units=256, fc2_units=256):
        """Initialize parameters and build model.
        Params
        ======
            input_size (int): number of dimensions for input layer
            seed (int): random seed
            fc1_units (int): number of nodes in the first hidden layer
            fc2_units (int): number of nodes in the second hidden layer
        """
        super(CriticModel, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights with near zero values."""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states, actions):
        """Build a critic network that maps (states, actions) pairs to Q-values."""
        xs = torch.cat((states, actions), dim=1)
        x = F.relu(self.fc1(xs))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Actor_Critic_Models():
    """
    Create object containing all models required per DDPG agent:
    local and target actor and local and target critic
    """

    def __init__(self, n_agents, state_size=24, action_size=2, seed=0):
        """
        Params
        ======
            n_agents (int): number of agents
            state_size (int): number of state dimensions for a single agent
            action_size (int): number of action dimensions for a single agent
            seed (int): random seed
        """
        self.actor_local = ActorModel(state_size, action_size, seed).to(device)
        self.actor_target = ActorModel(state_size, action_size, seed).to(device)
        critic_input_size = (state_size+action_size)*n_agents
        self.critic_local = CriticModel(critic_input_size, seed).to(device)
        self.critic_target = CriticModel(critic_input_size, seed).to(device)