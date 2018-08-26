import numpy as np
import gym
from argparse import ArgumentParser

def get_space_size(space):
    if type(space) is gym.spaces.Box:
        # TODO: Handle multi-dimensions boxes
        return space.shape[0]
    else:
        return space.n

def main():
    parser = ArgumentParser()
    parser.add_argument("--num_eps", help="number of episodes to learn", default=1000, type=int)
    parser.add_argument("--sum_steps", help="after how many steps a summary will be printed", default=200, type=int)
    parser.add_argument("environment", help="name of the gym environment to train on")
    args = parser.parse_args()

    experiments = {
        "FrozenLake-v0": frozen_lake_exp,
        "CartPole-v0": cart_pole_exp
    }

    total_rewards, episode_lengths = experiments.get(args.environment)(args)

    print(f"Overall mean score: {np.mean(total_rewards)} overall mean episode length: {np.mean(episode_lengths)}")

def frozen_lake_exp(args):
    from discrete_policy import discrete_policy
    from monte_carlo_control import monte_carlo_control

    env = gym.make(args.environment)

    state_size = get_space_size(env.observation_space)
    action_size = get_space_size(env.action_space)

    def save(i_epsiode):
        agent.save_checkpoint(f"./checkpoints/{args.environment}.ckpt", global_step=i_epsiode)

    agent = discrete_policy(state_size=state_size, action_size=action_size, learning_rate=.1, discount_factor=0.99)

    epsilon = 0.2
    sample = lambda x : agent.sample_eps_greedy(x, epsilon)

    return monte_carlo_control(env, args.num_eps, args.sum_steps, sample, agent.train, save)

def cart_pole_exp(args):
    from policy_network import policy_network
    from monte_carlo_control import monte_carlo_control

    env = gym.make(args.environment)

    state_size = get_space_size(env.observation_space)
    action_size = get_space_size(env.action_space)

    def save(i_epsiode):
        agent.save_checkpoint(f"./checkpoints/{args.environment}.ckpt", global_step=i_epsiode)

    agent = policy_network(state_size=state_size, action_size=action_size, hidden_units=10, learning_rate=.01, discount_factor=0.99)

    return monte_carlo_control(env, args.num_eps, args.sum_steps, agent.sample, agent.train, save)

if __name__ == '__main__':
    main()