import numpy as np
import tensorflow as tf

class agent():
    def __init__(self, sess, environment, discount_factor=0.95, epsilon=0.1, learning_rate=0.1):
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.sess = sess
        self.env = environment

        # Feed forward pass of the model
        self._inputs1 = tf.placeholder(shape=[1,16], dtype=tf.float32)
        W = tf.Variable(tf.random_uniform([16,4], 0, 0.01))
        self._Qout = tf.matmul(self._inputs1, W)
        self._predict = tf.argmax(self._Qout, 1)
        # Obtain the loss and update model with gradient descent
        self._nextQ = tf.placeholder(shape=[1,4], dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(self._nextQ - self._Qout))
        trainer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        self._updateModel = trainer.minimize(loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def predict(self, state):
        # Choose an action epsilon-greedy
        if np.random.rand(1) < self.epsilon:
            action = self.env.action_space.sample()
        else:
            Q_values = self.sess.run(self._predict, feed_dict={self._inputs1: np.identity(16)[state:state+1]})
            action = Q_values[0]
        return action

    def update_policy(self, old_state, action, new_state, reward, i_epsiode):
        # Obtain new Q values
        targetQ = self.sess.run(self._Qout, feed_dict={self._inputs1: np.identity(16)[old_state:old_state+1]})
        Q1 = self.sess.run(self._Qout, feed_dict={self._inputs1: np.identity(16)[new_state:new_state+1]})
        targetQ[0, action] = reward + self.discount_factor * np.max(Q1)

        # Train network using target and predicted Q values
        self.sess.run(self._updateModel, feed_dict={self._inputs1: np.identity(16)[old_state:old_state+1], self._nextQ: targetQ})

        # Decay epsilon
        self.epsilon = 1./((i_epsiode/50) + 10)

def monte_carlo_control(env, num_episodes, policy, update_policy):
    total_rewards = []
    episode_lengths = []
    for i_epsiode in range(1, num_episodes + 1):
        # Print out current episode for debugging
        if i_epsiode % 1000 == 0:
            print(f"Episode {i_epsiode}/{num_episodes} mean reward {np.mean(total_rewards[-1000:]):.3F} mean episode length {np.mean(episode_lengths[-1000:]):.3F}")

        state = env.reset()
        total_reward = 0
        done = False
        length = 0
        while not done: 
            length += 1

            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            update_policy(state, action, next_state, reward, i_epsiode)

            total_reward += reward
            state = next_state

        total_rewards.append(total_reward)
        episode_lengths.append(length)

    return total_rewards, episode_lengths

def main():
    num_episodes = 10000

    import gym
    env = gym.make('FrozenLake-v0')    

    tf.reset_default_graph()
    with tf.Session() as sess:
        a = agent(sess, env)
        total_rewards, episode_lengths = monte_carlo_control(env, num_episodes, a.predict, a.update_policy)

    print(f"Score over time: {sum(total_rewards) / num_episodes}")

if __name__ == '__main__':
    main()