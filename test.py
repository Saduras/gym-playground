from argparse import ArgumentParser
import gym
from policy_network import policy_network

def main():
    parser = ArgumentParser()
    parser.add_argument("--eps", help="number of episodes to learn", default=1000, type=int)
    parser.add_argument("environment", help="name of the gym environment to train on")
    args = parser.parse_args()

    env = gym.make(args.environment)   

    agent = policy_network(state_size=4, action_size=2, hidden_units=10, learning_rate=.01, discount_factor=0.99)
    agent.load_checkpoint(f"./checkpoints/{args.environment}.ckpt-{args.eps}")

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