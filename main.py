from pysc2.env import sc2_env
from pysc2.lib import features
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
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
	observation = timesteps[0].observation['screen'][_PLAYER_RELATIVE].reshape((1,64,64))
	step_count = 0
	
	for agent in agents:
		agent.reset()
	try:
		while True:
			total_frames += 1
			actions = []
			targets = []
			for agent in agents:
				out = agent.step(timesteps[0],observation.reshape((1,1,64,64)))
				actions.append(out[0])
				targets.append(out[1])
			timesteps = env.step(actions)
			next_observation = timesteps[0].observation['screen'][_PLAYER_RELATIVE].reshape((1,64,64))
			reward = timesteps[0].reward
			done = timesteps[0].last()
			if done:
				break

			for agent, action in zip(agents, targets):
				agent.train(observation, action, reward, next_observation)

			observation = next_observation
			step_count += 1

	except KeyboardInterrupt:
		return timesteps
		
	elapsed_time = time.time() - start_time
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
