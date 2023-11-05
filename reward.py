import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Reward:
    def __init__(self, observation_size, action_size):
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

    def learn(self, observation, action, )


        
