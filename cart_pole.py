import gym
import numpy as np

def run_episode(env, parameters):
    observation = env.reset()
    total_reward = 0
    for t in range(200):
        env.render()
        action = 0 if np.matmul(parameters, observation)<0 else 1
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            print(f"Episode finished after {t+1} timesteps")
            break

    return total_reward

# hill climbing
def train(submit):
    env = gym.make('CartPole-v0')

    noise_scale = 0.1
    parameters = np.random.rand(4) * 2 - 1
    best_reward = 0

    for i_episode in range(2000):
        new_params = parameters + (np.random.rand(4) * 2 - 1) * noise_scale
        reward = run_episode(env, new_params)
        print(f"episode {i_episode} reward {reward} best {best_reward}")
        print(parameters)
        if reward > best_reward:
            best_reward = reward
            parameters = new_params
            if reward >= 200:
                break

    return best_reward

r = train(submit=False)
print(r)