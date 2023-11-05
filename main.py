import os
import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.animation import FuncAnimation

from movingdot import MovingDotEnv
# from QLearningAgent import QLearningAgent
from ACAgent import ActorCriticAgent

# def main_QLearning():
#     env = MovingDotEnv()
#     agent = QLearningAgent(env)
#     state = env.reset()

#     def update(frame):
#         nonlocal state 
#         ax.clear()
#         #print(f"Current state: {state}")

#         if np.array_equal(state, [5, 5]):
#             env.render(ax)
#             plt.pause(0.1)
#             ani.event_source.stop()
#             return
        
#         action = agent.select_action(state)
#         next_state, reward, done, _ = env.step(action)
#         agent.learn(state, action, reward, next_state, done)
        
#         if not done:
#             state = next_state
#             env.render(ax)
#             plt.pause(0.1)
#         else:
#             env.render(ax)
#             plt.pause(0.1)
#             ani.event_source.stop()
#             return
            
#     fig, ax = plt.subplots()
#     ani = FuncAnimation(fig, update, frames=1000, repeat=False)
#     plt.show()
#     update()



def main_AC():
    env = MovingDotEnv()
    state_size = np.prod(np.array(env.observation_space.shape))
    action_size = env.action_space.n
    agent = ActorCriticAgent(state_size, action_size)
    #agent = ActorCriticAgent_LSTM(state_size, action_size)

    state = env.reset()

    def update(frame):
        nonlocal state
        ax.clear()

        if np.array_equal(state, [5, 5]):
            env.render(ax)
            plt.pause(0.1)
            ani.event_source.stop()
            return

        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, action, reward, next_state, done)

        if not done:
            state = next_state
            env.render(ax)
            plt.pause(0.1)
        else:
            env.render(ax)
            plt.pause(0.1)
            ani.event_source.stop()
            return


    fig, ax = plt.subplots()
    ani = FuncAnimation(fig, update, frames=1000, repeat=False)
    plt.show()
    #update()







#main_QLearning()
main_AC()


