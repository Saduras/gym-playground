import numpy as np
import gym
from argparse import ArgumentParser

from policy_network import policy_network
from monte_carlo_control import monte_carlo_control

def get_space_size(space):
    if type(space) is gym.spaces.Box:
        # TODO: Handle multi-dimensions boxes
        return space.shape[0]
    else:
        return space.n

def main():
    parser = ArgumentParser()
    parser.add_argument("--num_eps", help="number of episodes to learn", default=1000, type=int)
    parser.add_argument("environment", help="name of the gym environment to train on")
    args = parser.parse_args()

    env = gym.make(args.environment)
    
    print(env.observation_space)
    print(env.action_space)

    state_size = get_space_size(env.observation_space)
    action_size = get_space_size(env.action_space)

    agent = policy_network(state_size=state_size, action_size=action_size, hidden_units=10, learning_rate=.01, discount_factor=0.99)

    def save(i_epsiode):
        agent.save_checkpoint(f"./checkpoints/{args.environment}.ckpt", global_step=i_epsiode)

    total_rewards, episode_lengths = monte_carlo_control(env, args.num_eps, agent.sample, agent.train, save)

    print(f"Overall mean score: {np.mean(total_rewards)} overall mean episode length: {np.mean(episode_lengths)}")

if __name__ == '__main__':
    main()