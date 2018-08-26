import numpy as np
import gym

def monte_carlo_control(env, num_episodes, sum_steps, policy, update_policy, save_checkpoint):
    total_rewards = []
    episode_lengths = []

    for i_epsiode in range(1, num_episodes + 1):
        # Print out current episode for debugging
        if i_epsiode % sum_steps == 0:
            print(f"Episode {i_epsiode}/{num_episodes} mean reward {np.mean(total_rewards[-sum_steps:]):.3F} mean episode length {np.mean(episode_lengths[-sum_steps:]):.3F}")
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

        update_policy(episode)

        total_rewards.append(total_reward)
        episode_lengths.append(length)

    return total_rewards, episode_lengths