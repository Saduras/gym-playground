import numpy as np
import sys

def monte_carlo_control(env, num_episodes, discount_factor=0.95, epsilon=0.1, learning_rate=0.8):
    Q = np.zeros([env.observation_space.n, env.action_space.n])

    def policy(obs, i): 
        return np.argmax(Q[obs,:] + np.random.randn(1, env.action_space.n)*(1./(i+1)))

    total_rewards = []
    for i_epsiode in range(1, num_episodes + 1):
        # Print out current episode for debugging
        if i_epsiode % 1000 == 0:
            print(f"Episode {i_epsiode}/{num_episodes} mean reward {np.mean(total_rewards[-1000:])}")
            sys.stdout.flush()

        state = env.reset()
        total_reward = 0
        done = False
        while not done: 
            action = policy(state, i_epsiode)
            next_state, reward, done, _ = env.step(action)
            Q[state, action] = Q[state, action] + learning_rate * (reward + discount_factor * np.max(Q[next_state,:]) - Q[state, action])
            total_reward += reward
            state = next_state

        total_rewards.append(total_reward)

    return total_rewards, policy

def main():
    num_episodes = 2000

    import gym
    env = gym.make('FrozenLake-v0')
    total_rewards, policy = monte_carlo_control(env, num_episodes, learning_rate=0.8, discount_factor=0.95)

    print(f"Score over time: {sum(total_rewards) / num_episodes}")

if __name__ == '__main__':
    main()