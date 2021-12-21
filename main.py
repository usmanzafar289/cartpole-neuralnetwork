import random
import sys
import os
import gym
from cellular_automata_agent import CellularAutomataAgent as Agent

MAX_EPISODES_STEPS = 10000

def main():
    env = gym.make('CartPole-v0')
    env._max_episode_steps = MAX_EPISODES_STEPS

    agent = Agent(env)
    agent.run()
    env.close()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        try:
            exit()
        except SystemExit:
            os._exit(0)