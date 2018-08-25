from argparse import ArgumentParser
import gym
import numpy as np
from policy_network import load_network

def main():
    parser = ArgumentParser()
    parser.add_argument("--eps", help="number of episodes to learn", default=1000, type=int)
    parser.add_argument("environment", help="name of the gym environment to train on")
    args = parser.parse_args()

    env = gym.make(args.environment)
    obs_discrete = type(env.observation_space) is gym.spaces.Discrete   

    agent = load_network(f"./checkpoints/{args.environment}.ckpt", args.eps)

    num_episodes = 10

    for i_episode in range(1, num_episodes+1):
        total_reward = 0
        done = False

        state = env.reset()
        if obs_discrete:
            state = np.identity(env.observation_space.n)[state:state+1]

        while not done: 
            env.render()
            action = agent.sample(state)
            state, reward, done, _ = env.step(action)
            if obs_discrete:
                state = np.identity(env.observation_space.n)[state:state+1]
            total_reward += reward

        print(f"Episode {i_episode} finished with {total_reward}")

if __name__ == '__main__':
    main()