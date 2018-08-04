import numpy as np
import tensorflow as tf

def monte_carlo_control(env, num_episodes, discount_factor=0.95, epsilon=0.1, learning_rate=0.8):
    
    tf.reset_default_graph()

    # Feed forward pass of the model
    inputs1 = tf.placeholder(shape=[1,16], dtype=tf.float32)
    W = tf.Variable(tf.random_uniform([16,4], 0, 0.01))
    Qout = tf.matmul(inputs1, W)
    predict = tf.argmax(Qout, 1)
    # Obtain the loss and update model with gradient descent
    nextQ = tf.placeholder(shape=[1,4], dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(nextQ - Qout))
    trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    updateModel = trainer.minimize(loss)

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        total_rewards = []
        episode_lengths = []
        for i_epsiode in range(1, num_episodes + 1):
            # Print out current episode for debugging
            if i_epsiode % 1000 == 0:
                print(f"Episode {i_epsiode}/{num_episodes} mean reward {np.mean(total_rewards[-1000:])}")

            state = env.reset()
            total_reward = 0
            done = False
            length = 0
            while not done: 
                length += 1

                # Choose an action epsilon-greedy
                action, allQ = sess.run([predict, Qout], feed_dict={inputs1: np.identity(16)[state:state+1]})
                if np.random.rand(1) < epsilon:
                    action[0] = env.action_space.sample()

                next_state, reward, done, _ = env.step(action[0])

                # Obtain new Q values
                Q1 = sess.run(Qout, feed_dict={inputs1: np.identity(16)[next_state:next_state+1]})
                maxQ1 = np.max(Q1)
                targetQ = allQ
                targetQ[0, action[0]] = reward + discount_factor * maxQ1

                # Train network using target and predicted Q values
                _, W1 = sess.run([updateModel, W], feed_dict={inputs1: np.identity(16)[state:state+1], nextQ: targetQ})

                total_reward += reward
                state = next_state

            # Decay epsilon
            epsilon = 1./((i_epsiode/50) + 10)

            total_rewards.append(total_reward)
            episode_lengths.append(length)

    return total_rewards, episode_lengths

def main():
    num_episodes = 10000

    import gym
    env = gym.make('FrozenLake-v0')
    total_rewards, episode_lengths = monte_carlo_control(env, num_episodes, learning_rate=0.8, discount_factor=0.95)

    print(f"Score over time: {sum(total_rewards) / num_episodes}")

if __name__ == '__main__':
    main()