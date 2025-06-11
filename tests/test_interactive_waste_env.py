#! /usr/bin/env python
import numpy as np
import random

from dl_envs.toxic_waste.toxic_waste_env_v2 import Actions, ToxicWasteEnvV2, ProblemType
from dl_envs.toxic_waste.heuristic_agents.greedy_agent import GreedyAgent
from typing import List, Tuple
from pathlib import Path

RNG_SEED = 12072023
N_CYCLES = 100
ACTION_MAP = {'w': Actions.UP, 's': Actions.DOWN, 'a': Actions.LEFT, 'd': Actions.RIGHT, 'q': Actions.STAY, 'e': Actions.INTERACT}
np.set_printoptions(precision=3, linewidth=2000, threshold=1000)


def get_model_obs(raw_obs) -> Tuple[np.ndarray, np.ndarray]:
	conv_obs = []
	arr_obs = []

	if isinstance(raw_obs[0], dict):
		conv_obs = raw_obs[0]['conv']
		arr_obs = np.array(raw_obs[0]['array'])
	else:
		conv_obs = raw_obs[0][0].reshape(1, *raw_obs[0].shape)
		arr_obs = raw_obs[0][1:]
	return conv_obs.reshape(1, *conv_obs.shape), arr_obs


def main():
	
	field_size = (15, 15)
	layout = 'cramped_room'
	n_players = 2
	has_slip = False
	n_objects = 3
	max_episode_steps = 500
	facing = True
	layer_obs = True
	centered_obs = False
	use_render = True
	use_frames = False
	render_mode = ['human', 'rgb_array'] if use_frames else ['human']
	problem_type = ProblemType.MOVE_CATCH
	s_problem_type = 'move_catch'
	only_movement = False
	data_dir = Path(__file__).parent.absolute().parent.absolute() / 'data'
	
	env = ToxicWasteEnvV2(field_size, layout, n_players, n_objects, max_episode_steps, RNG_SEED, data_dir, facing, centered_obs, render_mode, use_render,
						  slip=has_slip, is_train=True, pick_all=False, problem_type=problem_type)
	env.seed(RNG_SEED)
	obs, *_ = env.reset()
	# print(env.get_filled_field())

	waste_order = [0, 1, 2]
	random.shuffle(waste_order)
	print(waste_order)
	greedy_agents = [GreedyAgent(player.position, player.orientation, player.name, dict([(idx, env.objects[idx].position) for idx in range(n_objects)]),
	                             123456789, env.field, 2, env.door_pos, agent_type=player.agent_type) for player in env.players if player.name == 'human']
	for model in greedy_agents:
		model.waste_order = waste_order.copy()

	env.render()
	
	for i in range(N_CYCLES):

		print('Iteration: %d' % (i + 1))
		# actions = [np.random.choice(range(6)) for _ in range(n_players)]
		actions = []
		print('\n'.join(['Player %s at (%d, %d) with orientation (%d, %d)' % (env.players[idx].name, *env.players[idx].position, *env.players[idx].orientation)
			   for idx in range(n_players)]))
		greedy_count = 0
		for idx in range(n_players):
			if env.players[idx].name != 'human':
				valid_action = False
				while not valid_action:
					human_input = input("Action for agent %s:\t" % env.players[idx].name)
					try:
						action = int(ACTION_MAP[human_input])
						if action < len(ACTION_MAP):
							valid_action = True
							actions.append(action)
						else:
							print('Action ID must be between 0 and %d, you gave ID %d' % (len(ACTION_MAP), action))
					except KeyError as e:
						print('Key error caught: %s' % str(e))
			else:
				actions.append(greedy_agents[greedy_count].act(env.create_observation(), only_movement, s_problem_type))
		print(' '.join([Actions(action).name for action in actions]))
		print(env.objects)
		state, rewards, dones, _, info = env.step(actions)
		print(env.objects)
		print(state)
		# print(env.get_filled_field())
		if use_frames:
			obs = env.render()
		if dones:
			obs, *_ = env.reset()
			greedy_agents = [GreedyAgent(player.position, player.orientation, player.name, dict([(idx, env.objects[idx].position) for idx in range(n_objects)]),
			                             123456789, env.field, 2, env.door_pos, agent_type=player.agent_type) for player in env.players if player.name == 'human']
			random.shuffle(waste_order)
			for model in greedy_agents:
				model.waste_order = waste_order
			if use_frames:
				obs = env.render()
			env.render()

		obs_shape = obs.shape if not isinstance(obs, list) else obs[0]['conv'].shape
		print(obs.reshape((1, *obs_shape)).shape if not isinstance(obs, list) else obs[0]['conv'].reshape((1, *obs_shape[1:], obs_shape[0])).shape, obs)

	env.close()


if __name__ == '__main__':
	main()
