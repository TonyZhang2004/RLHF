import numpy as np

class QLearningAgent:
    def __init__(self, env):
        self.env = env
        self.q_table = np.zeros((11, 11, env.action_space.n))
        self.learning_rate = 0.1
        self.discount_factor = 0.99
        self.exploration_rate = 1.0
        self.exploration_decay = 0.995
        self.min_exploration_rate = 0.2

    def select_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[int(state[0]), int(state[1]), :])

    def learn(self, state, action, reward, next_state, done):
        best_next_action = np.argmax(self.q_table[int(next_state[0]), int(next_state[1]), :])
        td_target = reward + self.discount_factor * self.q_table[int(next_state[0]), int(next_state[1]), best_next_action]
        td_error = td_target - self.q_table[int(state[0]), int(state[1]), action]
        self.q_table[int(state[0]), int(state[1]), action] += self.learning_rate * td_error

        if not done:
            self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)
