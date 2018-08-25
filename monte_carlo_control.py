import numpy as np
import gym
from argparse import ArgumentParser

def monte_carlo_control(env, num_episodes, policy, update_policy, save_checkpoint):
    total_rewards = []
    episode_lengths = []
    for i_epsiode in range(1, num_episodes + 1):
        # Print out current episode for debugging
        if i_epsiode % 1000 == 0:
            print(f"Episode {i_epsiode}/{num_episodes} mean reward {np.mean(total_rewards[-1000:]):.3F} mean episode length {np.mean(episode_lengths[-1000:]):.3F}")
            save_checkpoint(i_epsiode)

        state = env.reset()
        total_reward = 0
        done = False
        length = 0
        episode = []
        while not done: 
            length += 1

            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, next_state, reward))
            total_reward += reward
            state = next_state

        update_policy(episode, i_epsiode)

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

    from cart_pole_agent import cart_pole_agent
    agent = cart_pole_agent(state_size=state_size, action_size=action_size, hidden_units=10, learning_rate=.01, discount_factor=0.99)

    def save(i_epsiode):
        agent.save_checkpoint(f"./checkpoints/{args.environment}.ckpt", global_step=i_epsiode)

    total_rewards, episode_lengths = monte_carlo_control(env, args.num_eps, agent.sample, agent.train, save)

    print(f"Overall mean score: {np.mean(total_rewards)} overall mean episode length: {np.mean(episode_lengths)}")

if __name__ == '__main__':
    main()