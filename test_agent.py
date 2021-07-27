"""Script to test the trained agent in the same environment.

@author: Rohith Banka


The saved DNN weights(from train_agent.py) are loaded into another Network for testing purposes.
"""



# Import all the required modules and libraries.
import torch
import time
import random
import sys
import numpy as np
from dqn_agent import Agent
from unityagents import UnityEnvironment
from train_agent import get_environment_info, get_agent


# Number of episodes to test.
test_episodes=5



def test_agent_in_env(env, agent, brain_name, test_episodes):
	'''
	To test the agent in the environment.

	agent: DQN Agent.
	brain_name: In this case, the first brain available, and is set as the default brain to learn policies.
	'''
	for i_episode in range(1, test_episodes+1):

		# get the environment information.
		# Important to set the train_mode to False during inference/testing.
		env_info = env.reset(train_mode=False)[brain_name]

		# Initial state of the environment.
		state = env_info.vector_observations[0]

		# Initial score is set to 0.
		score = 0

		while True:

			# Agent to take the action in the given state.
			action = agent.act(state)

			# To get the environment information after the action.
			env_info = env.step(action)[brain_name]

			# get the state from the environment, post action.
			next_state = env_info.vector_observations[0]

			# get the reward for action.
			reward = env_info.rewards[0]
			done = env_info.local_done[0]

			state = next_state

			score += reward


			if done:
				break

		print(f"Episode: {i_episode}, the agent has been able to collect {round(score, 3)} yellow bananas")

	env.close()
	print("Done with testing the agent, So closing the environment...")



def main():
	location = sys.argv[1]
	dqn_weights = sys.argv[2]
	dqn_type = sys.argv[3]
	env, brain_name, brain, action_size, state_size = get_environment_info(location)
	agent = get_agent(state_size=state_size, action_size=action_size, dqn_type=dqn_type)
	try:
		agent.network.load_state_dict(torch.load(dqn_weights))
		print("Loaded the weights successfully...")
	except Exception as e:
		raise Exception(f"{e}")

	test_agent_in_env(env, agent, brain_name, test_episodes)


if __name__ == '__main__':
	main()