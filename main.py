from pysc2.env import sc2_env
from pysc2.lib import features
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_SELECTED = features.SCREEN_FEATURES.selected.index

import numpy as np
import time, sys
import agentmanager
import gflags as flags
FLAGS = flags.FLAGS
argv = FLAGS(sys.argv)

def run_loop(agents, env):
	total_frames = 0
	start_time = time.time()
	
	action_spec = env.action_spec()
	observation_spec = env.observation_spec()
	for agent in agents:
		agent.setup(observation_spec, action_spec)
		
	timesteps = env.reset()
	observation = timesteps[0].observation['screen'][_PLAYER_RELATIVE].reshape((1,1,64,64))
	observation = np.concatenate((observation, timesteps[0].observation['screen'][_SELECTED].reshape((1,1,64,64))), axis=1)
	selected = timesteps[0].observation['screen'][_SELECTED].reshape((1,1,64,64))
	step_count = 0
	
	for agent in agents:
		agent.reset()
	total_reward = 0
	try:
		while True:
			total_frames += 1
			actions = []
			targets = []
			for agent in agents:
				out = agent.step(timesteps[0],observation.reshape((1,2,64,64)),selected)
				targets.append(out[0])
				if not out[1] == None:
					actions.append(out[1])
			
			selected = timesteps[0].observation['screen'][_SELECTED].reshape((1,1,64,64))
			timesteps = env.step(targets)
			reward = timesteps[0].reward
			total_reward += reward
			done = timesteps[0].last()
			if done:
				next_observation = None
				for agent, action in zip(agents, actions):
					agent.train(observation, action, reward, next_observation)
					agent.reset()
				break
			next_observation = timesteps[0].observation['screen'][_PLAYER_RELATIVE].reshape((1,1,64,64))
			next_observation = np.concatenate((next_observation, selected), axis=1)
			for agent, action in zip(agents, actions):
				agent.train(observation, action, reward, next_observation)

			observation = next_observation
			step_count += 1
			
	except KeyboardInterrupt:
		return timesteps
		
	elapsed_time = time.time() - start_time
	print("Finished with score {}".format(total_reward))
	print("Took %.3f seconds for %s steps: %.3f fps" % (elapsed_time, total_frames, 
														total_frames / elapsed_time))
	return timesteps

if __name__=='__main__':
	num_episodes = 10000
	for episode in range(num_episodes):
		with sc2_env.SC2Env(
                    "MoveToBeacon",
                    agent_race='T',
                    bot_race='P',
                    difficulty="1",
                    visualize=True) as env:
			agent = agentmanager.Agent()
			run_result = run_loop([agent], env)
			reward = run_result[0].reward
			env.close()

		print("%d'th game result :"%episode,reward)
