import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from movingdot import MovingDotEnv
from ACAgent import ActorCriticAgent

patience = 10




def simulate(display, agent, env):    
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
        rewards = 0
        losses = []
        for _ in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            losses.append(agent.learn(state, action, reward, next_state, done))

            rewards += reward

            if not done:
                state = next_state
            else:
                break
        
        return rewards, (sum(losses) / len(losses))

    if display:
        fig, ax = plt.subplots()
        ani = FuncAnimation(fig, update, frames=max_steps, repeat=False)
        plt.show()
    else:
        return run()
        



def main():
    env = MovingDotEnv()
    state_size = np.prod(np.array(env.observation_space.shape))
    action_size = env.action_space.n
    agent = ActorCriticAgent(state_size, action_size)
    agent.load()

    best_reward = float('-inf')
    prev_reward = 0
    sum = 0

    for iteration in range(10_000):
        env.reset()
        rewards, losses = simulate(False, agent, env)
        if best_reward < rewards:
            best_reward = rewards
            print("saving...")
            agent.save()

        if iteration > 50:
            if prev_reward > rewards:
                sum += 1
            else:      
                prev_reward = rewards
                sum = 0

            if sum >= patience:
                break
            
        print(iteration, "total rewards =", rewards, "| average loss =", losses[0])
    print("finished")

main()