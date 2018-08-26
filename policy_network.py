import numpy as np
import tensorflow as tf
import yaml

def load_network(path, global_step):
    with open(f"{path}.params", "r") as file:
        params = yaml.load(file)
    pn = policy_network(params["state_size"], params["action_size"], params["hidden_units"], params["discount_factor"], params["learning_rate"])
    pn.load_checkpoint(path, global_step)
    return pn

class policy_network():
    def __init__(self, state_size, action_size, hidden_units, discount_factor=0.95, learning_rate=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_units = hidden_units
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.sess = tf.InteractiveSession()

        # Feed forward pass of the model
        self.observations = tf.placeholder(tf.float32, shape=[None, state_size], name="observations")

        hidden_layer = tf.layers.dense(
            self.observations, 
            units=hidden_units, 
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer()
        )
        self.output = tf.layers.dense(
            hidden_layer,
            units=action_size,
            activation=tf.nn.softmax,
            kernel_initializer=tf.contrib.layers.xavier_initializer()
        )

        # Backwards pass
        self.sampled_actions = tf.placeholder(tf.int32, shape=[None, 1], name="sampled_actions")
        self.advantages = tf.placeholder(tf.float32, shape=[None, 1], name="reward_signal")

        # Convert list of action indicies into one hot vectors
        one_hot_sampled = tf.one_hot(tf.reshape(self.sampled_actions,shape=[-1]) , self.action_size)
        
        self.loss = tf.losses.log_loss(
            labels=one_hot_sampled,
            predictions=self.output,
            weights=self.advantages
        )
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)

        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()

    def load_checkpoint(self, path, global_step):
        path = f"{path}-{global_step}"
        print(f"Loading checkpoint: {path}")
        self.saver.restore(self.sess, path)

    def save_checkpoint(self, path, global_step):
        self.saver.save(self.sess, path, global_step=global_step)

        with open(f"{path}.params", "w") as file:
            yaml.dump({
              "state_size": self.state_size,
              "action_size": self.action_size,
              "hidden_units": self.hidden_units,
              "discount_factor": self.discount_factor,
              "learning_rate": self.learning_rate
            }, file)

    def sample(self, state):
        # Use policy output to sample action probabilistic
        state = state.reshape([1, -1])
        a_dist = self.sess.run(self.output, feed_dict={self.observations: state})[0]
        action = np.random.choice(self.action_size, p=a_dist)
        return action

    def sample_eps_greedy(self, state, epsilon):
        if np.random.uniform() < epsilon:
            action = np.random.choice(range(self.action_size))
        else:
            state = state.reshape([1, -1])
            a_dist = self.sess.run(self.output, feed_dict={self.observations: state})[0]
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
        states = np.vstack(episode_array[:, 0])
        actions = np.vstack(episode_array[:, 1])

        rewards = np.vstack(self.discount_rewards(episode_array[:, 3]))
        rewards -= rewards.mean()
        if rewards.std() != 0:
            rewards /= rewards.std()

        feed_dict = {
            self.observations: states,
            self.sampled_actions: actions,
            self.advantages: rewards
        }
        self.sess.run(self.train_op, feed_dict)