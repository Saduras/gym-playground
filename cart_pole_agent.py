import numpy as np
import tensorflow as tf

### Source: https://github.com/breeko/Simple-Reinforcement-Learning-with-Tensorflow/blob/master/Part%202%20-%20Policy-based%20Agents.ipynb
class cart_pole_agent():
    def __init__(self, sess, environment, state_size, action_size, hidden_units, discount_factor=0.95, epsilon=0.1, learning_rate=0.1, batch_size=5):
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.sess = sess
        self.env = environment
        self.state_size = state_size
        self.batch_size = batch_size

        # Feed forward pass of the model
        self.observations = tf.placeholder(tf.float32, shape=[None, state_size], name="input_x")
        
        W1 = tf.get_variable("W1", shape=[state_size, hidden_units], initializer=tf.contrib.layers.xavier_initializer())
        layer1 = tf.nn.relu(tf.matmul(self.observations, W1))

        W2 = tf.get_variable("W2", shape=[hidden_units, 1], initializer=tf.contrib.layers.xavier_initializer())
        self.output = tf.nn.sigmoid(tf.matmul(layer1, W2))

        # Backwards pass
        trainable_vars = [W1, W2]
        self.input_y = tf.placeholder(tf.float32, shape=[None,1], name="input_y")
        self.advantages = tf.placeholder(tf.float32, shape=[None], name="reward_signal")

        # Loss function
        log_lik = tf.log(self.input_y * (self.input_y - self.output) + (1 - self.input_y) * (self.input_y + self.output))
        loss = -tf.reduce_mean(log_lik * self.advantages)

        # Gradients
        self.new_grads = tf.gradients(loss, trainable_vars)
        self.W1_grad = tf.placeholder(tf.float32, name="batch_grad1")
        self.W2_grad = tf.placeholder(tf.float32, name="batch_grad2")

        self.gradients = np.array([np.zeros(var.get_shape()) for var in trainable_vars])

        # Learning
        batch_grad = [self.W1_grad, self.W2_grad]
        adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.update_grads = adam.apply_gradients(zip(batch_grad, [W1, W2]))

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def predict(self, state):
        # Choose an action epsilon-greedy
        if np.random.rand(1) < self.epsilon:
            action = self.env.action_space.sample()
        else:
            state = np.reshape(state, [1, self.state_size])
            a_dist = self.sess.run(self.output, feed_dict={self.observations: state})
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

    def update_policy(self, episode, i_epsiode):
        episode_array = np.array(episode)
        old_states = np.stack(episode_array[:,0], axis=0)
        actions = episode_array[:,1].reshape(-1,1)

        rewards = self.discount_rewards(episode_array[:,3])
        rewards -= rewards.mean()
        rewards /= rewards.std()

        self.gradients += np.array(self.sess.run(self.new_grads, feed_dict={
            self.observations: old_states, 
            self.input_y: actions, 
            self.advantages: rewards
        }))
        if i_epsiode % self.batch_size == 0:
            self.sess.run(self.update_grads, feed_dict={self.W1_grad: self.gradients[0], self.W2_grad: self.gradients[1]})
            self.gradients *= 0

        # Decay epsilon
        self.epsilon = 1./((i_epsiode/50) + 10)