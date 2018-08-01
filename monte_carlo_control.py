import numpy as np
import sys
from sklearn.linear_model import LinearRegression

def epsilon_greedy_policy(Q, observation, epsilon, nA):
    # equal probability for all actions
    probs = np.ones(nA, dtype=float) * epsilon / nA
    # find best action
    best_action = np.argmax(Q(observation))
    # increase the probability of the best action to 1 - epsilon
    probs[best_action] += (1.0 - epsilon)
    return np.random.choice(nA, p=probs)

def lin_Q(observation, parameters):
    return np.matmul(observation.transpose(), parameters)

def run_episode(env, policy, render=False):
    episode = []
    state = env.reset()
    total_reward = 0
    done = False
    while not done: 
        action = policy(state)
        if render:
            env.render()
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        episode.append((state, action, reward))
        state = next_state
    return episode, total_reward
    
def monte_carlo_control(env, num_episodes, discount_factor=1.0, epsilon=0.1, learning_rate=0.1):
    # Init params for Q function
    q_params = np.random.rand(env.observation_space.shape[0], env.action_space.n)

    # initialize Q function with random training
    Q = LinearRegression()
    rand_X = np.array(np.random.rand(1, env.observation_space.shape[0]))
    rand_y = np.array(np.random.rand(1, env.action_space.n))
    print(rand_X.shape)
    Q.fit(rand_X, rand_y)

    def policy(obs, params): 
        return epsilon_greedy_policy(lambda x: Q.predict([x]), obs, epsilon=epsilon, nA=env.action_space.n)
    
    X_states = []
    y_rewards = []

    total_rewards = []
    for i_epsiode in range(1, num_episodes + 1):
        # Print out current episode for debugging
        if i_epsiode % 1000 == 0:
            print(f"Episode {i_epsiode}/{num_episodes} mean reward {np.mean(total_rewards[-1000:])}")
            sys.stdout.flush()

        # Generate an episode
        # An episode is an array of (state, action, reward) tuples
        episode, total_reward = run_episode(env, lambda x: policy(x, q_params))

        total_rewards.append(total_reward)
        
        # Update Q function
        for state, action, _ in episode:
            # Find the first occurance of the (state, action) pair in the episode
            first_occurence_idx = next(i for i,x in enumerate(episode)
                                       if x[0].all() == state.all() and x[1] == action)
            # Sum up all rewards since the first occurance
            G = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurence_idx:])])

            y = Q.predict([state])
            y[0][action] = G
            X_states.append(state)
            y_rewards.append(y[0])

        Q.fit(np.array(X_states), np.array(y_rewards))
        # print(f"score {Q.score(X_states, y_rewards)}")

    return q_params, policy

def main():
    observation = np.array([0.1, 0.2, 0.3, 0.4])
    q_parameters = np.array([[1, 0], [1, 0], [1, 0], [0, 0]])
    print(f"q_param.shape {q_parameters.shape}")
    print(f"Q.shape {lin_Q(observation, q_parameters).shape}")
    
    import gym
    env = gym.make('CartPole-v0')
    q_params, policy = monte_carlo_control(env, 20000, learning_rate=0.001)

if __name__ == '__main__':
    main()