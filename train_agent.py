import torch
import time
import random
import numpy as np
from collections import deque
from dqn_agent import *
from model import *
from unityagents import UnityEnvironment
import sys




# training parameters
num_episodes=2000
epsilon=1.0
epsilon_min=0.05
epsilon_decay=0.99
scores = []
scores_average_window=100
solved_score = 14

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_envionment_info(location):
    env = UnityEnvironment(location)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    action_size = brain.vector_action_space_size
    state_size = brain.vector_observation_space_size

    return (env, brain_name, brain, action_size, state_size)


def get_agent(state_size, action_size, dqn_type='DQN'):
    agent = Agent(state_size=state_size, action_size=action_size, dqn_type='DQN')
    return agent


def train_agent(env, brain_name, brain, action_size, state_size, agent, epsilon, num_episodes):

    print(brain_name)
    for i_episode in range(1, num_episodes+1):
        print(f"training episode: {i_episode}")
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]

        score = 0

        while True:
            action = agent.act(state, epsilon)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

            if done:
                break

        scores.append(score)
        average_score = np.mean(scores[i_episode - min(i_episode, scores_average_window): i_episode+1])

        epsilon = max(epsilon_min, epsilon_decay*epsilon)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, average_score), end="")

        # Print average score every scores_average_window episodes
        if i_episode % scores_average_window == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, average_score))
        
        # Check to see if the task is solved (i.e,. avearge_score > solved_score). 
        # If yes, save the network weights and scores and end training.
        if average_score >= solved_score:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, average_score))


            start_time = time.strftime("%Y%m%d-%H%M%S")
            network_name = "DQN_model_weights_"+start_time + ".pth"
            torch.save(agent.network.state_dict(), network_name)


            scores_filename = "DQN_agent_scores_"+start_time + ".csv"
            np.savetxt(scores_filename, scores, delimiter=",")
            break


    env.close()
    print("Closed the environment")





def main():
    location = sys.argv[1]
    env, brain_name, brain, action_size, state_size = get_envionment_info(location)
    agent = get_agent(state_size, action_size)
    train_agent(env, brain_name, brain, action_size, state_size, agent, epsilon, num_episodes)


if __name__ == "__main__":
    main()


