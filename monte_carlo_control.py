import numpy as np
import gym

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