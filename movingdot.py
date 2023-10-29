import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import QLearningAgent

class MovingDotEnv(gym.Env):
    def __init__(self):
        super(MovingDotEnv, self).__init__()
        self.action_space = spaces.Discrete(4)  # 0: Up, 1: Down, 2: Left, 3: Right
        self.observation_space = spaces.Box(low=0, high=10, shape=(2,), dtype=np.float32)
        self.state = None
        self.reset()

    def reset(self):
        self.state = np.array([5.0, 5.0])
        return self.state

    def step(self, action):
        if action == 0 and self.state[1] < self.observation_space.high:
            self.state[1] += 1
        elif action == 1 and self.state[1] > self.observation_space.low:
            self.state[1] -= 1
        elif action == 2 and self.state[0] < self.observation_space.high:
            self.state[0] -= 1
        elif action == 3 and self.state[0] < self.observation_space.low:
            self.state[0] += 1
        else:
            raise Exception("Invalid Action")
        
        self.state = np.clip(self.state, 0, 10)

        reward = -1
        done = False
        
        if np.array_equal(self.state, np.array([5.0, 5.0])):
            done = True
            reward = 10

        info = {}
        return self.state, reward, done, info

    def render(self, ax):
        ax.scatter(self.state[0], self.state[1], c='red')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

def get_user_action():
    print("Select action: 0: Up, 1: Down, 2: Left, 3: Right")
    action = int(input())
    return action

def get_user_reward():
    print("Provide a reward signal (positive or negative number):")
    reward = float(input())
    return reward

def get_user_done():
    print("Is the episode done? (yes/no)")
    done = input().strip().lower() == 'yes'
    return done


