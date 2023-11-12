from movingdot import MovingDotEnv
from ACAgent import ActorCriticAgent
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def simulate(agent, env):    
    state = env.reset()
    max_steps = 100
    
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
    
    fig, ax = plt.subplots()
    ani = FuncAnimation(fig, update, frames=max_steps, repeat=False)
    plt.show()


env = MovingDotEnv()
state_size = np.prod(np.array(env.observation_space.shape))
action_size = env.action_space.n
agent = ActorCriticAgent(state_size, action_size)
agent.load()

simulate(agent, env)

