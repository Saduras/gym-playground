import numpy as np
import yaml

def load_network(path, global_step):
    with open(f"{path}.params", "r") as file:
        params = yaml.load(file)
    pn = discrete_policy(params["state_size"], params["action_size"], params["discount_factor"], params["learning_rate"])
    pn.load_checkpoint(path, global_step)
    return pn

class discrete_policy():
    def __init__(self, state_size, action_size, discount_factor=0.95, learning_rate=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

        self.Q = np.zeros([state_size, action_size])

    def load_checkpoint(self, path, global_step):
        path = f"{path}-{global_step}"
        print(f"Loading checkpoint: {path}")
        
        with open(path, "r") as file:
            self.Q = yaml.load(file)["Q"]

    def save_checkpoint(self, path, global_step):
        with open(path, "w") as file:
            yaml.dump({"Q": self.Q}, file)

        with open(f"{path}.params", "w") as file:
            yaml.dump({
              "state_size": self.state_size,
              "action_size": self.action_size,
              "discount_factor": self.discount_factor,
              "learning_rate": self.learning_rate
            }, file)

    def sample_eps_greedy(self, state, epsilon):
        if np.random.uniform() < epsilon:
            action = int(np.random.choice(range(self.action_size)))
        else:
            a_dist = self.Q[state,:]
            action = np.argmax(a_dist)
        return action

    def discount_rewards(self, rewards):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def train(self, episode):
        episode_array = np.array(episode)
        states = episode_array[:, 0]
        actions = episode_array[:, 1]

        rewards = self.discount_rewards(episode_array[:, 3])
        rewards -= rewards.mean()
        if rewards.std() != 0:
            rewards /= rewards.std()

        for s, a, r in zip(states, actions, rewards):
            s = int(s)
            a = int(a)
            self.Q[s,a] = self.Q[s,a] + self.learning_rate * (r + np.max(self.Q[s,:]) - self.Q[s,a])
