def main():
    import gym
    env = gym.make('CartPole-v0')   

    from cart_pole_agent import cart_pole_agent
    agent = cart_pole_agent(state_size=4, action_size=2, hidden_units=10, learning_rate=.01, discount_factor=0.99)
    agent.load_checkpoint("./checkpoints/cart_pole.ckpt-10000")

    num_episodes = 10

    for i_episode in range(1, num_episodes+1):
        total_reward = 0
        done = False
        state = env.reset()
        while not done: 
            env.render()
            action = agent.sample(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state

        print(f"Episode {i_episode} finished with {total_reward}")

if __name__ == '__main__':
    main()