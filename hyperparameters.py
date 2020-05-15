# HYPERPARAMS:
# EPSILON = 1.0           # epsilon for the noise process added to the actions
# EPSILON_DECAY = 1e-6    # decay for epsilon above
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 0.001              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 20       # how often to update the network
UPDATE_TIMES = 10      # how many times to update the network each time

EPSILON = 0.8           # epsilon for the noise process added to the actions
MIN_EPSILON = 0.1
EPSILON_DECAY = 1e-6    # decay for epsilon above
NOISE_START = 1.0
NOISE_DECAY = 1.0
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")