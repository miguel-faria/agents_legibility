#! /usr/bin/env python

import sys
import argparse
import wandb
import numpy as np
import threading
import jax
import flax.linen as nn
import os
import random
import time
import yaml
import logging
import json
import traceback

from dl_algos.dqn import EPS_TYPE, DQNetwork
from dl_algos.single_model_madqn import CentralizedMADQN
from dl_envs.toxic_waste.toxic_waste_env_v1 import ToxicWasteEnvV1
from dl_envs.toxic_waste.toxic_waste_env_v2 import ToxicWasteEnvV2, Actions, ProblemType
from dl_envs.toxic_waste.heuristic_agents.greedy_agent import GreedyAgent
from pathlib import Path
from itertools import product
from typing import List, Union, Dict, Tuple, Optional
from datetime import datetime
from itertools import permutations
from wandb.wandb_run import Run

RNG_SEED = 21062023
ROBOT_NAME = 'astro'
INTERACTIVE_SESSION = False
ANNEAL_DECAY = 0.999
RESTART_WARMUP = 5
MOVE_PENALTY = -1
FINISH_REWARD = 100


def convert_joint_act(action: int, num_agents: int, n_actions: int) -> List[int]:
	actions_map = list(product(range(n_actions), repeat=num_agents))
	return np.array(actions_map[action])


def get_history_entry(obs: ToxicWasteEnvV2.Observation, actions: List[int], n_agents: int) -> List:
	entry = []
	for a_idx in range(n_agents):
		agent = obs.players[a_idx]
		action = actions[a_idx]
		entry += [agent.id, agent.position, agent.orientation, agent.held_objects, action]

	return entry


def input_callback(env: Union[ToxicWasteEnvV1, ToxicWasteEnvV2], stop_flag: threading.Event):
	try:
		while not stop_flag.is_set():
			command = input('Interactive commands:\n\trender - display renderization of the interaction\n\tstop_render - stops the renderization\nCommand: ')
			if command == 'render':
				env.use_render = True
			elif command == 'stop_render':
				if env.use_render:
					env.use_render = False

	except KeyboardInterrupt as ki:
		return


def model_execution(dqn_model: DQNetwork, eps: float, greedy_actions: bool, n_agents: int, n_joint_actions: int, v2_obs: Tuple, rng_gen: np.random.Generator,
                    waste_env: Union[ToxicWasteEnvV1, ToxicWasteEnvV2], episode_q_vals: List) -> List[int]:
	if rng_gen.random() < eps:
		actions = waste_env.action_space.sample()
	else:
		q_values = dqn_model.q_network.apply(dqn_model.online_state.params, v2_obs[0], v2_obs[1].reshape((1, 1)), rngs={"dropout": jax.random.PRNGKey(42)})[0]

		if greedy_actions:
			action = q_values.argmax(axis=-1)
		else:
			pol = np.isclose(q_values, q_values.max(), rtol=1e-10, atol=1e-10).astype(int)
			pol = pol / pol.sum()
			action = rng_gen.choice(range(n_joint_actions), p=pol)
		joint_action = int(jax.device_get(action))
		actions = convert_joint_act(joint_action, n_agents, waste_env.action_space[0].n)
		episode_q_vals.append(float(q_values[int(joint_action)]))

	return actions


def heuristic_execution(waste_env: Union[ToxicWasteEnvV1, ToxicWasteEnvV2], agent_models: List[GreedyAgent], train_only_movement: bool = False) -> List[int]:
	actions = []
	obs = waste_env.create_observation()
	for model in agent_models:
		actions.append(model.act(obs, train_only_movement))

	return actions


def train_astro_model(waste_env: ToxicWasteEnvV2, waste_model: CentralizedMADQN, agent_models: List[GreedyAgent], waste_order: List,
                      num_iterations: int, max_timesteps: int, batch_size: int, optim_learn_rate: float, tau: float, initial_eps: float, final_eps: float,
                      eps_type: str, rng_seed: int, logger: logging.Logger, exploration_decay: float = 0.99, warmup: int = 0, target_freq: int = 1000,
                      train_freq: int = 10, greedy_actions: bool = True, debug_mode: bool = False, use_frames: bool = False,
                      anneal_cool: float = 0.9, use_tracker: bool = False, performance_tracker: Optional[Run] = None, tracker_panel: str = '') -> List:
	history = []
	random.seed(rng_seed)
	np.random.seed(rng_seed)
	rng_gen = np.random.default_rng(rng_seed)
	n_agents = waste_env.n_players
	n_joint_actions = waste_env.action_space[0].n * waste_env.n_players

	obs, *_ = waste_env.reset()
	if use_frames:
		obs = waste_env.render()
	else:
		obs = obs[0]['conv']
	obs_shape = obs.shape
	if waste_env.use_render:
		waste_env.render()
	dqn_model = waste_model.madqn
	if dqn_model.cnn_layer:
		if use_frames:
			obs_reshaped = obs.reshape((1, *obs_shape))
		else:
			obs_reshaped = obs.reshape((1, *obs_shape[1:], obs_shape[0]))
		dqn_model.init_network_states(rng_seed, obs_reshaped, optim_learn_rate)
	else:
		dqn_model.init_network_states(rng_seed, obs, optim_learn_rate)

	start_time = time.time()
	epoch = 0
	start_record_it = 0
	avg_episode_len = []

	for it in range(num_iterations):
		logger.info("Iteration %d out of %d" % (it + 1, num_iterations))
		done = False
		episode_rewards = 0
		episode_q_vals = 0
		episode_start = epoch
		avg_loss = []
		while not done:
			# interact with environment
			if eps_type == 'epoch':
				eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, epoch, max_timesteps)
			else:
				eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, it, num_iterations)
			if rng_gen.random() < eps:
				actions = waste_env.action_space.sample()
			else:
				dqn_model = waste_model.madqn
				if dqn_model.cnn_layer:
					if use_frames:
						obs_reshaped = obs.reshape((1, *obs_shape))
					else:
						obs_reshaped = obs.reshape((1, *obs_shape[1:], obs_shape[0]))
					q_values = dqn_model.q_network.apply(dqn_model.online_state.params, obs_reshaped)[0]
				else:
					q_values = dqn_model.q_network.apply(dqn_model.online_state.params, obs)

				if greedy_actions:
					joint_action = q_values.argmax(axis=-1)
				else:
					pol = np.isclose(q_values, q_values.max(), rtol=1e-10, atol=1e-10).astype(int)
					pol = pol / pol.sum()
					joint_action = rng_gen.choice(range(n_joint_actions), p=pol)
				actions = convert_joint_act(joint_action, n_agents, waste_env.action_space[0].n)
				episode_q_vals += float(q_values[int(joint_action)])

			if debug_mode:
				logger.info('Environment current state')
				logger.info(waste_env.get_full_env_log())
				logger.info('Player actions: %s' % str([Actions(act).name for act in actions]))

			next_obs, rewards, terminated, timeout, infos = waste_env.step(actions)
			if use_frames:
				next_obs = waste_env.render()
			else:
				next_obs = next_obs[0]['conv']

			if waste_env.use_render:
				waste_env.render()
			if debug_mode:
				logger.info('Player rewards: %s' % str(rewards))

			step_reward = sum(rewards) / waste_model.num_agents
			episode_rewards += step_reward

			if terminated:
				finished = np.ones(n_agents)
			else:
				finished = np.zeros(n_agents)

			# store new samples
			if dqn_model.cnn_layer:
				if use_frames:
					obs_reshaped = obs.reshape((1, *obs_shape))
					next_obs_reshaped = next_obs.reshape((1, *obs_shape))
				else:
					obs_reshaped = obs.reshape((1, *obs_shape[1:], obs_shape[0]))
					next_obs_reshaped = next_obs.reshape((1, *obs_shape[1:], obs_shape[0]))
				waste_model.replay_buffer.add(obs_reshaped, next_obs_reshaped, np.array(actions), rewards, finished, [infos])
			else:
				waste_model.replay_buffer.add(obs, next_obs, np.array(actions), rewards, finished, [infos])

			# update Q-network and target network
			if epoch >= warmup:
				if epoch % train_freq == 0:
					loss = jax.device_get(waste_model.update_model(batch_size))
					avg_loss += [loss]

				if epoch % target_freq == 0:
					waste_model.madqn.update_target_model(tau)

			obs = next_obs
			epoch += 1
			sys.stdout.flush()
			if terminated or timeout:
				episode_len = epoch - episode_start
				avg_episode_len += [episode_len]
				if use_tracker:
					performance_tracker.log({
							tracker_panel + "-charts/performance/mean_episode_q_vals": episode_q_vals / episode_len,
							tracker_panel + "-charts/performance/mean_episode_return": episode_rewards / episode_len,
							tracker_panel + "-charts/performance/episodic_length":     episode_len,
							tracker_panel + "-charts/performance/avg_episode_length":  np.mean(avg_episode_len),
							tracker_panel + "-charts/control/iteration":               it + 1,
							tracker_panel + "-charts/control/exploration":             eps,
							tracker_panel + "-charts/losses/td_loss":                  sum(avg_loss) / max(len(avg_loss), 1)},
							step=(it + start_record_it)
					)
				obs, *_ = waste_env.reset()
				if waste_env.use_render:
					waste_env.render()
				episode_rewards = 0
				episode_q_vals = 0
				episode_start = epoch
				avg_loss = []
				done = True
				[model.reset(waste_order, dict([(idx, waste_env.objects[idx].position) for idx in range(waste_env.n_objects)])) for model in agent_models]


def main():
	parser = argparse.ArgumentParser(description='Train DQN model for Astro waste disposal game.')

	# Multi-agent DQN params
	parser.add_argument('--nagents', dest='n_agents', type=int, required=True, help='Number of agents in the environment')
	parser.add_argument('--architecture', dest='architecture', type=str, required=True, help='DQN architecture to use from the architectures yaml')
	parser.add_argument('--buffer', dest='buffer_size', type=int, required=True, help='Size of the replay buffer in the DQN')
	parser.add_argument('--gamma', dest='gamma', type=float, required=False, default=0.99, help='Discount factor for agent\'s future rewards')
	parser.add_argument('--gpu', dest='use_gpu', action='store_true', help='Flag that signals the use of gpu for the training')
	parser.add_argument('--ddqn', dest='use_ddqn', action='store_true', help='Flag that signals the use of a Double DQN')
	parser.add_argument('--cnn', dest='use_cnn', action='store_true', help='Flag that signals the use of a CNN as entry for the DQN architecture')
	parser.add_argument('--dueling', dest='dueling_dqn', action='store_true', help='Flag that signals the use of a Dueling DQN architecture')
	parser.add_argument('--tensorboard', dest='use_tensorboard', action='store_true',
	                    help='Flag the signals the use of a tensorboard summary writer. Expects argument --tensorboardDetails to be present')
	parser.add_argument('--tensorboardDetails', dest='tensorboard_details', nargs='+', required=False, default=None,
	                    help='List with the details for the tensorboard summary writer: <log_dirname: str>, <queue_size :int>, <flush_time: int>, <suffix: str>'
	                         ' Use only in combination with --tensorboard option')

	# Train parameters
	# parser.add_argument('--cycles', dest='n_cycles', type=int, required=True,
	# 					help='Number of training cycles, each cycle spawns the field with a different food items configurations.')
	parser.add_argument('--iterations', dest='n_iterations', type=int, required=True, help='Number of iterations to run training')
	parser.add_argument('--batch', dest='batch_size', type=int, required=True, help='Number of samples in each training batch')
	parser.add_argument('--train-freq', dest='train_freq', type=int, required=True, help='Number of epochs between each training update')
	parser.add_argument('--target-freq', dest='target_freq', type=int, required=True, help='Number of epochs between updates to target network')
	parser.add_argument('--alpha', dest='learn_rate', type=float, required=False, default=2.5e-4, help='Learn rate for DQN\'s Q network')
	parser.add_argument('--tau', dest='target_learn_rate', type=float, required=False, default=2.5e-6, help='Learn rate for the target network')
	parser.add_argument('--init-eps', dest='initial_eps', type=float, required=False, default=1., help='Exploration rate when training starts')
	parser.add_argument('--final-eps', dest='final_eps', type=float, required=False, default=0.05, help='Minimum exploration rate for training')
	parser.add_argument('--eps-decay', dest='eps_decay', type=float, required=False, default=0.5, help='Decay rate for the exploration update')
	parser.add_argument('--cycle-eps-decay', dest='cycle_eps_decay', type=float, required=False, default=0.95, help='Decay rate for the exploration update')
	parser.add_argument('--eps-type', dest='eps_type', type=str, required=False, default='log', choices=['linear', 'exp', 'log', 'epoch'],
	                    help='Type of exploration rate update to use: linear, exponential (exp), logarithmic (log), epoch based (epoch)')
	parser.add_argument('--warmup-steps', dest='warmup', type=int, required=False, default=10000, help='Number of epochs to pass before training starts')
	parser.add_argument('--tensorboard-freq', dest='tensorboard_freq', type=int, required=False, default=1,
	                    help='Number of epochs between each log in tensorboard. Use only in combination with --tensorboard option')
	parser.add_argument('--checkpoint-freq', dest='checkpoint_freq', type=int, required=False, default=10,
	                    help='Number of epochs between each model train checkpointing.')
	parser.add_argument('--restart', dest='restart_train', action='store_true',
	                    help='Flag that signals that train is suppose to restart from a previously saved point.')
	parser.add_argument('--debug', dest='debug', action='store_true', help='Flag signalling debug mode for model training')
	parser.add_argument('--fraction', dest='fraction', type=str, default='0.5', help='Fraction of JAX memory pre-compilation')
	parser.add_argument('--anneal-decay', dest='anneal_decay', type=float, default=ANNEAL_DECAY, help='Decay value for the heuristic annealing')
	parser.add_argument('--initial-temp', dest='init_temp', type=float, default=1.0, help='Initial value for the annealing temperature.')
	parser.add_argument('--models-dir', dest='models_dir', type=str, default='', help='Directory to store trained models, if left blank stored in default location')
	parser.add_argument('--logs-dir', dest='logs_dir', type=str, default='', help='Directory to store logs, if left blank stored in default location')
	parser.add_argument('--data-dir', dest='data_dir', type=str, default='',
	                    help='Directory to retrieve data regarding configs and model performances, if left blank using default location')
	parser.add_argument('--checkpoint-file', dest='checkpoint_file', type=str, required=False, default='', help='File with data from previous training checkpoint')
	parser.add_argument('--buffer-smart-add', dest='buffer_smart_add', action='store_true',
	                    help='Flag denoting the use of smart sample add to experience replay buffer instead of first-in first-out')
	parser.add_argument('--buffer-method', dest='buffer_method', type=str, required=False, default='uniform', choices=['uniform', 'weighted'],
	                    help='Method of deciding how to add new experience samples when replay buffer is full')
	parser.add_argument('--use-curriculum', dest='use_curriculum', action='store_true',
	                    help='Flag that signals training using previously trained models as a starting model')
	parser.add_argument('--curriculum-model', dest='curriculum_model', type=str, default='', help='Path to model to use as a starting model to improve.')
	parser.add_argument('--problem-type', dest='problem_type', type=str, choices=['only_movement', 'move_catch', 'pick_one', 'only_green', 'green_yellow', 'all_balls', 'full'],
	                    help='Different types of problem simplification')

	parser.add_argument('--train-only-movement', dest='only_movement', action='store_true', help='Flag denoting train only of moving in environment')
	parser.add_argument('--has-pick-all', dest='has_pick_all', action='store_true', help='Flag denoting all green and yellow balls have to be picked before human exiting')

	# Environment parameters
	parser.add_argument('--version', dest='env_version', type=int, required=True, help='Environment version to use')
	parser.add_argument('--game-levels', dest='game_levels', type=str, required=True, nargs='+', help='Level to train Astro in.')
	parser.add_argument('--max-env-steps', dest='max_steps', type=int, required=True, help='Maximum number of steps for environment timeout')
	parser.add_argument('--field-size', dest='field_size', type=int, required=True, nargs='+', help='Number of rows and cols in field')
	parser.add_argument('--slip', dest='has_slip', action='store_true', help='')
	parser.add_argument('--require_facing', dest='require_facing', action='store_true', help='')
	parser.add_argument('--agent-centered', dest='centered_obs', action='store_true', help='')
	parser.add_argument('--use-encoding', dest='use_encoding', action='store_true', help='')
	parser.add_argument('--layer-obs', dest='use_layers', action='store_true', help='Environment observation in layer organization')
	parser.add_argument('--render', dest='use_render', action='store_true', help='Flag signaling the use of a render')
	parser.add_argument('--render-mode', dest='render_mode', type=str, nargs='+', required=False, default=None,
	                    help='List of render modes for the environment')

	args = parser.parse_args()
	# DQN args
	n_agents = args.n_agents
	architecture = args.architecture
	buffer_size = args.buffer_size
	gamma = args.gamma
	use_gpu = args.use_gpu
	dueling_dqn = args.dueling_dqn
	use_ddqn = args.use_ddqn
	use_cnn = args.use_cnn
	use_tensorboard = args.use_tensorboard
	# [log_dir: str, queue_size: int, flush_interval: int, filename_suffix: str]
	tensorboard_details = args.tensorboard_details

	# Train args
	n_iterations = args.n_iterations
	batch_size = args.batch_size
	train_freq = args.train_freq
	target_freq = args.target_freq
	learn_rate = args.learn_rate
	target_update_rate = args.target_learn_rate
	initial_eps = args.initial_eps
	final_eps = args.final_eps
	eps_decay = args.eps_decay
	eps_type = args.eps_type
	warmup = args.warmup
	tensorboard_freq = args.tensorboard_freq
	checkpoint_freq = args.checkpoint_freq
	debug = args.debug
	decay_anneal = args.anneal_decay
	anneal_temp = args.init_temp
	chkpt_file = args.checkpoint_file
	use_curriculum = args.use_curriculum
	curriculum_model = args.curriculum_model
	problem_type = args.problem_type
	only_movement = args.only_movement

	# Astro environment args
	env_version = args.env_version
	game_levels = args.game_levels
	field_size = tuple(args.field_size) if len(args.field_size) == 2 else tuple([args.field_size[0], args.field_size[0]])
	has_slip = args.has_slip
	max_episode_steps = args.max_steps
	facing = args.require_facing
	centered_obs = args.centered_obs
	use_encoding = args.use_encoding
	render_mode = args.render_mode
	use_render = args.use_render

	os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = args.fraction
	if not use_gpu:
		jax.default_device(jax.devices("cpu")[0])

	now = datetime.now()
	home_dir = Path(__file__).parent.absolute().parent.absolute()
	log_dir = Path(args.logs_dir) if args.logs_dir != '' else home_dir / 'logs'
	data_dir = Path(args.data_dir) if args.data_dir != '' else home_dir / 'data'
	models_dir = Path(args.models_dir) / 'models' if args.models_dir != '' else home_dir / 'models'
	configs_dir = data_dir / 'configs'
	model_path = models_dir / 'astro_disposal_dqn' / now.strftime("%Y%m%d-%H%M%S")
	rng_gen = np.random.default_rng(RNG_SEED)

	if chkpt_file != '' and args.restart_train:
		with open(chkpt_file, 'r') as j_file:
			chkpt_data = json.load(j_file)
	else:
		chkpt_data = {}
		for level in game_levels:
			chkpt_data[level] = {'iteration': 0, 'temp': anneal_temp}
		chkpt_file = str(models_dir / ('v%d_train_checkpoint_data.json' % env_version))

	if use_curriculum:
		try:
			assert curriculum_model != ''
		except AssertionError:
			print('Attempt at using curriculum learning but doesn\'t supply a model to use as a starting point')
			return

	only_movement = True if problem_type == "only_movement" else False
	if problem_type == "only_movement":
		problem_code = ProblemType.ONLY_MOVE
	elif problem_type == "move_catch":
		problem_code = ProblemType.MOVE_CATCH
	elif problem_type == "pick_one":
		problem_code = ProblemType.PICK_ONE
	elif problem_type == "only_green":
		problem_code = ProblemType.ONLY_GREEN
	elif problem_type == "green_yellow":
		problem_code = ProblemType.GREEN_YELLOW
	elif problem_type == "all_balls":
		problem_code = ProblemType.BALLS_ONLY
	else:
		problem_code = ProblemType.FULL

	with open(configs_dir / 'q_network_architectures.yaml') as architecture_file:
		arch_data = yaml.safe_load(architecture_file)
		if architecture in arch_data.keys():
			n_layers = arch_data[architecture]['n_layers']
			layer_sizes = arch_data[architecture]['layer_sizes']
			n_conv_layers = arch_data[architecture]['n_cnn_layers']
			cnn_size = arch_data[architecture]['cnn_size']
			cnn_kernel = [tuple(elem) for elem in arch_data[architecture]['cnn_kernel']]
			cnn_strides = arch_data[architecture]['cnn_strides']
			pool_window = [tuple(elem) for elem in arch_data[architecture]['pool_window']]
			pool_strides = arch_data[architecture]['pool_strides']
			pool_padding = arch_data[architecture]['pool_padding']
			cnn_properties = [n_conv_layers, cnn_size, cnn_kernel, cnn_strides, pool_window, pool_strides, pool_padding]

	run = wandb.init(project='trustat-toxic-waste', entity='ana-v-carrasco97-inesc-id',
	                 config={
			                 "agent_type":           "joint_policy",
			                 "env_version":          "v1" if env_version == 1 else "v2",
			                 "agents":               n_agents,
			                 "online_learing_rate":  learn_rate,
			                 "target_learning_rate": target_update_rate,
			                 "discount":             gamma,
			                 "eps_decay_type":       eps_type,
			                 "eps_decay":            eps_decay,
			                 "iterations":           n_iterations,
			                 "buffer_size":          buffer_size,
			                 "buffer_add":           "smart" if args.buffer_smart_add else "plain",
			                 "buffer_add_method":    args.buffer_method if args.buffer_smart_add else "fifo",
			                 "batch_size":           batch_size,
			                 "online_frequency":     train_freq,
			                 "target_frequency":     target_freq,
			                 "architecture":         architecture,
			                 "problem":              problem_type
	                 },
	                 dir=tensorboard_details[0],
	                 name=('joint_policy_' + now.strftime("%Y%m%d-%H%M%S")),
	                 sync_tensorboard=True)

	if len(logging.root.handlers) > 0:
		for handler in logging.root.handlers:
			logging.root.removeHandler(handler)

	# logging.basicConfig(filename=(log_dir / 'train_astro_disposal_multi_dqn_ctce.log'), level=logging.INFO, format='%(name)s %(asctime)s %(levelname)s:\t%(message)s')
	for game_level in game_levels:
		log_filename = ('train_astro_disposal_joint_model_%s' % game_level + '_' + now.strftime("%Y%m%d-%H%M%S"))
		logger = logging.getLogger("%s" % game_level)
		logger.setLevel(logging.INFO)
		file_handler = logging.FileHandler(log_dir / (log_filename + '.log'))
		file_handler.setFormatter(logging.Formatter('%(name)s %(asctime)s %(levelname)s:\t%(message)s'))
		file_handler.setLevel(logging.INFO)
		logger.addHandler(file_handler)
		Path.mkdir(model_path, parents=True, exist_ok=True)
		Path.mkdir(models_dir / 'checkpoints', parents=True, exist_ok=True)
		try:
			with open(configs_dir / 'layouts' / (game_level + '.yaml')) as config_file:
				objects = yaml.safe_load(config_file)['objects']
				n_objects = len(objects['unspecified']) if env_version == 1 else sum([len(objects[key]['ids']) for key in objects.keys() if key != 'unspecified'])

			logger.info('#######################################')
			logger.info('Starting Astro Waste Disposal DQN Train')
			logger.info('#######################################')
			logger.info('Level %s setup' % game_level)
			if env_version == 1:
				env = ToxicWasteEnvV1(field_size, game_level, n_agents, n_objects, max_episode_steps, RNG_SEED, facing, args.use_layers, centered_obs,
				                      use_encoding, render_mode, slip=has_slip, use_render=use_render, joint_obs=True)
			else:
				env = ToxicWasteEnvV2(field_size, game_level, n_agents, n_objects, max_episode_steps, RNG_SEED, data_dir, facing, centered_obs, render_mode,
				                      slip=has_slip, is_train=True, use_render=use_render, joint_obs=True, pick_all=args.has_pick_all, problem_type=problem_code)

			obs, *_ = env.reset(seed=RNG_SEED)

			logger.info('Getting human behaviour model')
			agent_models = []
			for player in env.players:
				if env_version == 1:
					agent_models.append(GreedyAgent(player.position, player.orientation, player.name,
					                                dict([(idx, env.objects[idx].position) for idx in range(n_objects)]), RNG_SEED, env.field, env_version,
					                                agent_type=player.agent_type))
				else:
					agent_models.append(GreedyAgent(player.position, player.orientation, player.name,
					                                dict([(idx, env.objects[idx].position) for idx in range(n_objects)]), RNG_SEED, env.field, env_version,
					                                env.door_pos, agent_type=player.agent_type))

			logger.info('Train setup')
			waste_idx = []
			for obj in env.objects:
				waste_idx.append(env.objects.index(obj))
			waste_seqs = list(permutations(waste_idx))
			waste_order = list(rng_gen.choice(np.array(waste_seqs)))
			for model in agent_models:
				model.waste_order = waste_order

			logger.info('Creating DQN and starting train')
			tensorboard_details[0] = tensorboard_details[0] + '/astro_disposal_' + game_level + '_' + now.strftime("%Y%m%d-%H%M%S")
			tensorboard_details += ['astro_' + game_level]
			start_it = chkpt_data[game_level]['iteration']
			start_temp = chkpt_data[game_level]['temp']
			if args.restart_train and curriculum_model != '':
				curriculum_model = 'v2_l-%s-checkpoint_ctce.model' % game_level
			astro_dqn = CentralizedMADQN(n_agents if not env.use_joint_obs else 1, env.action_space[0].n, n_layers, convert_joint_act, nn.relu, layer_sizes, buffer_size, gamma,
			                             env.action_space, env.observation_space, use_gpu, dueling_dqn, use_ddqn, use_cnn, (env_version == 2), False,
			                             use_tracker=use_tensorboard, tracker=run, cnn_properties=cnn_properties,
			                             buffer_data=(args.buffer_smart_add, args.buffer_method))

			train_astro_model(env, astro_dqn, agent_models, waste_order, n_iterations, max_episode_steps * n_iterations, batch_size, learn_rate,
			                  target_update_rate, initial_eps, final_eps, eps_type, RNG_SEED, logger, eps_decay, warmup, target_freq, train_freq,
			                  tensorboard_freq, debug_mode=debug, interactive=INTERACTIVE_SESSION)

			logger.info('Saving model and history list')
			astro_dqn.save_model(game_level, model_path, logger)

		except KeyboardInterrupt as ks:
			logger.info('Caught keyboard interrupt, cleaning up and closing.')
			wandb.finish()

		except Exception as e:
			logger.error("Caught an unexpected exception while training level %s: %s\n%s" % (game_level, str(e), traceback.format_exc()))

	wandb.finish()


if __name__ == '__main__':
	main()
