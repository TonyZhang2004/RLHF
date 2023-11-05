import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from movingdot import MovingDotEnv
from ACAgent import ActorCriticAgent



def simulate(display):
    env = MovingDotEnv()
    state_size = np.prod(np.array(env.observation_space.shape))
    action_size = env.action_space.n
    agent = ActorCriticAgent(state_size, action_size)

    state = env.reset()

    max_steps = 50

    def update(frame):
        nonlocal state
        ax.clear()

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
            plt.close()
            return
        
    def run():
        nonlocal state
        for _ in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)

            if not done:
                state = next_state
            else:
                return

    if display:
        fig, ax = plt.subplots()
        ani = FuncAnimation(fig, update, frames=max_steps, repeat=False)
        plt.show()
    else:
        run()
        



def main():
    for iteration in range(10_000):
        print(iteration)
        if iteration % 1_000 == 0:
            simulate(True)
        else:
            simulate(False)
    print("finished")
    simulate(True)

main()