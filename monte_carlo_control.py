import numpy as np
import gym
from argparse import ArgumentParser

def monte_carlo_control(env, num_episodes, policy, update_policy, save_checkpoint):
    total_rewards = []
    episode_lengths = []

    obs_discrete = type(env.observation_space) is gym.spaces.Discrete

    for i_epsiode in range(1, num_episodes + 1):
        # Print out current episode for debugging
        if i_epsiode % 1000 == 0:
            print(f"Episode {i_epsiode}/{num_episodes} mean reward {np.mean(total_rewards[-1000:]):.3F} mean episode length {np.mean(episode_lengths[-1000:]):.3F}")
            save_checkpoint(i_epsiode)

        state = env.reset()
        if obs_discrete:
            state = np.identity(env.observation_space.n)[state:state+1]

        total_reward = 0
        done = False
        length = 0
        episode = []
        while not done: 
            length += 1

            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            if obs_discrete:
                next_state = np.identity(env.observation_space.n)[next_state:next_state+1]

            episode.append((state, action, next_state, reward))
            total_reward += reward
            state = next_state

        update_policy(episode)

        total_rewards.append(total_reward)
        episode_lengths.append(length)

    return total_rewards, episode_lengths
    
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
    
    state_size = get_space_size(env.observation_space)
    action_size = get_space_size(env.action_space)

    from policy_network import policy_network
    agent = policy_network(state_size=state_size, action_size=action_size, hidden_units=10, learning_rate=.01, discount_factor=0.99)

    def save(i_epsiode):
        agent.save_checkpoint(f"./checkpoints/{args.environment}.ckpt", global_step=i_epsiode)

    total_rewards, episode_lengths = monte_carlo_control(env, args.num_eps, agent.sample, agent.train, save)

    print(f"Overall mean score: {np.mean(total_rewards)} overall mean episode length: {np.mean(episode_lengths)}")

if __name__ == '__main__':
    main()