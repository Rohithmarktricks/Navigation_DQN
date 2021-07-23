"""
train_agent.py: This is the source code to train the agent.
@author: Rohith Banka.
Initial code has been provided by the team by Udacity Deep Reinforcement Learning Team, 2021.

Unity ML agents have Academy and brains.

Academy: This element in Unity ML, orchestrates agents and their decision-making process.
Brain: We train the agent by optimizing the policy called "Brain". We control this brain using Python API.


For further information, Please refer to the following Medium article:
https://towardsdatascience.com/an-introduction-to-unity-ml-agents-6238452fcf4c

"""


# Import the required modules.
import torch
import time
import random
import numpy as np
from collections import deque
from dqn_agent import *
from model import *
from unityagents import UnityEnvironment
import sys


# training hyperparameters
num_episodes=2000
epsilon=1.0
epsilon_min=0.05
epsilon_decay=0.99
scores = []
scores_average_window=100
required_score = 14


def get_envionment_info(location):
    '''To get the information about the environment from the given location of Unity ML agent'''
    env = UnityEnvironment(location)
    # We check for the first brain available, and set it as the default brain, we will be controlling using Python API.
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # action size is the total number of distinct actions.
    action_size = brain.vector_action_space_size
    # state size is the total number of dimensions of the each state in the environment. In our case, it's 37.
    state_size = brain.vector_observation_space_size

    return (env, brain_name, brain, action_size, state_size)


def get_agent(state_size, action_size, dqn_type='DQN'):
    '''Initializes and returns the agent'''
    agent = Agent(state_size=state_size, action_size=action_size, dqn_type=dqn_type)
    print(f"Agent has been initialized with dqn_type as {dqn_type}...")
    return agent


def train_agent(env, brain_name, brain, action_size, state_size, agent, epsilon, dqn_type,num_episodes):
    '''Trains the agent
        
        Input parameters:
        brain_name: the name of the first brain available.
        brain: The policy that we train.
        action_size: Total number of actions that an agent can take.
        state_size: 37 dimension (in this navigation case)
        epsilon: This is used for epsilon-greedy action selection.
        num_episodes: Total number of episodes to train the agent.


    '''
    print(brain_name)

    '''
    1. Reset the training environment at the beginning of each episode.
    2. Get the current state i.e., s
    3. Use Epsilon-greedy policy to perform and action(a), in the environment in the given state (s)
    4. Get the reward and next_state of the environment for action (a)
    5. Calcuate the error between actual and expeted Q values for s(t), a(t), r(t), s(t+1), in turn this is used to train Neural Networks
    6. Update the total reward received and set s(t) <- s(t+1)
    7. Steps 1 to 3 will be repeated until the episode is done.


    However, in the case below the training process stops even if the total score of agent > 14.
    '''


    for i_episode in range(1, num_episodes+1):
        print(f"training episode: {i_episode}")
        env_info = env.reset(train_mode=True)[brain_name]

        # initial state
        state = env_info.vector_observations[0]

        # Initial score for each episode is 0.
        score = 0

        while True:
            # get the action.
            action = agent.act(state, epsilon)

            # take the action in the environment
            env_info = env.step(action)[brain_name]

            # now get the next state of the environment, post taking the action.
            next_state = env_info.vector_observations[0]

            # reward from the environment after action (a)
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

            if done:
                break


        # After every episode, append score to the scores list, calculate the mean.
        scores.append(score)

        # mean score is calcuated over present episode until i_episode > 100. 
        # (so past 100 scores will be taken for average if they available, else all the existing score will be used for mean)
        average_score = np.mean(scores[i_episode - min(i_episode, scores_average_window): i_episode+1])

        # epsilon value is being reduced as the agent is learning and action space can exploited.
        epsilon = max(epsilon_min, epsilon_decay*epsilon)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, average_score), end="")

        # Print average score every scores_average_window episodes
        if i_episode % scores_average_window == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, average_score))
        
        # Check to see if the task is solved (i.e,. avearge_score > required_score). 
        # If yes, save the network weights and scores and end training.
        if average_score >= required_score:
            print('\n{} Agent has learnt to solve the environment in {:d} episodes!\tAverage Score: {:.2f}'.format(dqn_type,i_episode, average_score))

            #  To save the weights of the Neural network
            start_time = time.strftime("%Y%m%d-%H%M%S")
            network_name = dqn_type+"model_weights_"+start_time + ".pth"
            torch.save(agent.network.state_dict(), network_name)
            print(f"Saved the {dqn_type} model weights at {network_name}")

            # To save the recorded Scores data.
            scores_filename = dqn_type+"_agent_scores_"+start_time + ".csv"
            np.savetxt(scores_filename, scores, delimiter=",")
            print(f"Scores of the training process for {dqn_type} model has been stored at {scores_filename}")
            break

    # close the environment after all the episodes.
    env.close()
    print("Closed the environment")





def main():
    '''Main function that takes the location/path of the environment'''
    location = sys.argv[1]

    dqn_type = sys.argv[2]
    # gets the environment information.
    env, brain_name, brain, action_size, state_size = get_envionment_info(location)

    # Initializes the agent with state_size, and action_size of the environment.
    agent = get_agent(state_size, action_size, dqn_type=dqn_type)

    # Train the agent.
    train_agent(env, brain_name, brain, action_size, state_size, agent, epsilon, dqn_type, num_episodes)


if __name__ == "__main__":
    main()


