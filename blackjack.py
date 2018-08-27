import numpy as np
import gym
from collections import defaultdict

def plot_value_function(V, title="Value Function"):
    import matplotlib
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    """
    Plots the value function as a surface plot.
    """
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))

def sample_eps_greedy(Q, state, epsilon):
    if np.random.uniform() < epsilon:
        action = int(np.random.choice(range(2)))
    else:
        a_dist = Q[state]
        action = np.argmax(a_dist)
    return action

# Implementation of SARSA (state-action-reward-state-action)
# also known as TD(0) (temporal difference)
def sarsa(env, num_episodes, sum_steps, lr, gamma, eps):
    total_rewards = []
    episode_lengths = []

    # Blackjack observation come in tuples (Discrete(32),Discrete(11),Discrete(2))
    # (Player current sum, dealers showing card, player holds an ace)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    actions = []

    for i_epsiode in range(1, num_episodes + 1):
        # Print out current episode for debugging
        if i_epsiode % sum_steps == 0:
            print(f"Episode {i_epsiode}/{num_episodes} mean reward {np.mean(total_rewards[-sum_steps:]):.3F} mean episode length {np.mean(episode_lengths[-sum_steps:]):.3F}")
            # print(f"Q table: {Q} \n\n")
            unique, counts = np.unique(actions, return_counts=True)
            print(f"actions: {dict(zip(unique, counts))}")

        state = env.reset()

        total_reward = 0
        
        done = False
        length = 0
        episode = []
        while not done: 
            length += 1

            action = sample_eps_greedy(Q, state, eps)
            next_state, reward, done, _ = env.step(action)

            episode.append((state, action, next_state, reward))

            Q[state][action] = Q[state][action] + lr * (reward + gamma * np.max(Q[next_state]) - Q[state][action])
            Q[state][action] = np.clip(Q[state][action], -1, 1)

            total_reward += reward
            actions.append(action)
            state = next_state

        total_rewards.append(total_reward)
        episode_lengths.append(length)

    # Plot value function
    V = defaultdict(float)
    for state, actions in Q.items():
        action_value = np.max(actions)
        V[state] = action_value
    plot_value_function(V, title="Optimal Value Function")

    return total_rewards, episode_lengths

def main():
    env_name = "Blackjack-v0"
    num_eps = 500000
    sum_steps = 50000

    env = gym.make(env_name)

    lr = 0.1            # learning rate
    gamma = 0.95        # discount factor
    eps = 0.1           # epsilon - chance to pick a random action

    total_rewards, episode_lengths = sarsa(env, num_eps, sum_steps, lr, gamma, eps)

    print(f"Overall mean score: {np.mean(total_rewards)} overall mean episode length: {np.mean(episode_lengths)}")

if __name__ == '__main__':
    main()