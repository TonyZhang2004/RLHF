import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.network(state)

class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, state):
        return self.network(state)

class ActorCriticAgent:
    def __init__(self, state_size, action_size, learning_rate_actor=1e-3, learning_rate_critic=1e-2, gamma=0.90, beta=0.1):
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=learning_rate_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=learning_rate_critic)
        self.gamma = gamma
        self.beta = beta

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        probabilities = self.actor(state)
        action = torch.multinomial(probabilities, 1).item()
        return action

    def learn(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)  
        next_state = torch.tensor(next_state, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        # Critic update
        value = self.critic(state)
        next_value = self.critic(next_state)
        td_target = reward + self.gamma * next_value * (1 - done)
        td_error = td_target - value
        self.optimizer_critic.zero_grad()
        td_error.backward()
        self.optimizer_critic.step()

        # Actor update
        probabilities = self.actor(state)
        distribution = torch.distributions.Categorical(probabilities)
        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()
        loss_actor = -(log_prob * td_error.detach() + self.beta * entropy)
        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        self.optimizer_actor.step()



