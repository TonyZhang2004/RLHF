import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Reward:
    def __init__(self, observation_size, action_size, learning_rate=1e-3, gamma=0.90, beta=0.1):
        self.network = nn.Sequential(
            nn.Linear(observation_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size),
            nn.Softmax(dim=-1)
        )

    def predict(self, observation, action):
        inp = observation + action
        return self.network(inp)

    
    

def basic_reward(state):
    target = np.array([5.0, 5.0])
    # distance = np.linalg.norm(state - target) ** 2
    # return 1 / (distance if distance != 0 else 1e-5)
    if np.array_equal(state, target):
        return 50
    return -1


        
