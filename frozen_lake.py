import numpy as np
import gym

def sample_eps_greedy(Q, state, epsilon):
    if np.random.uniform() < epsilon:
        action = int(np.random.choice(range(Q.shape[1])))
    else:
        a_dist = Q[state,:]
        action = np.argmax(a_dist)
    return action

# Implementation of SARSA (state-action-reward-state-action)
# also known as TD(0) (temporal difference)
def sarsa(env, num_episodes, sum_steps, lr, gamma, eps):
    total_rewards = []
    episode_lengths = []

    Q = np.zeros([env.observation_space.n, env.action_space.n])

    for i_epsiode in range(1, num_episodes + 1):
        # Print out current episode for debugging
        if i_epsiode % sum_steps == 0:
            print(f"Episode {i_epsiode}/{num_episodes} mean reward {np.mean(total_rewards[-sum_steps:]):.3F} mean episode length {np.mean(episode_lengths[-sum_steps:]):.3F}")

        state = env.reset()

        total_reward = 0
        done = False
        length = 0
        episode = []
        while not done: 
            length += 1

            action = sample_eps_greedy(Q,state, eps)
            next_state, reward, done, _ = env.step(action)

            episode.append((state, action, next_state, reward))

            Q[state,action] = Q[state,action] + lr * (reward + gamma * np.max(Q[next_state,:]) - Q[state,action])

            total_reward += reward
            state = next_state


        total_rewards.append(total_reward)
        episode_lengths.append(length)

    return total_rewards, episode_lengths

def main():
    env_name = "FrozenLake-v0"
    num_eps = 50000
    sum_steps = 1000

    env = gym.make(env_name)

    lr = 0.1            # learning rate
    gamma = 0.95        # discount factor
    eps = 0.1           # epsilon - chance to pick a random action

    total_rewards, episode_lengths = sarsa(env, num_eps, sum_steps, lr, gamma, eps)

    print(f"Overall mean score: {np.mean(total_rewards)} overall mean episode length: {np.mean(episode_lengths)}")

if __name__ == '__main__':
    main()