import os
import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.animation import FuncAnimation

from movingdot import MovingDotEnv
from QLearningAgent import QLearningAgent

def main() -> None :
  def update(frame):
        global state  
        ax.clear()
        print(f"Current state: {state}")
        
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, action, reward, next_state, done)
    
        state = next_state if not done else env.reset()  
    
        env.render(ax)
        plt.pause(0.1) 
    
        if done:
            ani.event_source.stop()
            
  env = MovingDotEnv()
  agent = QLearningAgent()
  state = env.reset()
  fig, ax = plt.subplots()
  ani = FuncAnimation(fig, update(), frames=100, repeat=False)
  
  plt.show()  


main()



        