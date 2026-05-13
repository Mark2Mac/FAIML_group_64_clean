"""Sample script for training a control policy on the Hopper environment

    Here you will implement the training loop for REINFORCE and A2C
"""
import gymnasium as gym

def main():
    env = gym.make('Hopper-v4')

    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space

    #TODO: implement training loop for REINFORCE and A2C using the agent defined in agent.py

if __name__ == '__main__':
    main()