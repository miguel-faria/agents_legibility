#! /usr/bin/env python

import time
import gymnasium
import numpy as np
import yaml

from env.toxic_waste_env_base import BaseToxicEnv, AgentType, HoldState, CellEntity
from pathlib import Path
from enum import IntEnum, Enum
from gymnasium.spaces import Box, MultiDiscrete
from typing import List, Tuple, Any, Union, Optional
from termcolor import colored
from copy import deepcopy
from collections import namedtuple
from itertools import product


MOVE_PENALTY = 0
HOLD_REWARD = 0.0
DELIVER_WASTE = 0
ROOM_CLEAN = 0
PICK_REWARD = 0
ADJ_REWARD = 0.0
IDENTIFY_REWARD = 0.0


class Actions(IntEnum):
	UP = 0
	DOWN = 1
	LEFT = 2
	RIGHT = 3
	INTERACT = 4
	STAY = 5
	# IDENTIFY = 6


class ActionDirection(Enum):
	UP = (-1, 0)
	DOWN = (1, 0)
	LEFT = (0, -1)
	RIGHT = (0, 1)
	INTERACT = (0, 0)
	STAY = (0, 0)
	# IDENTIFY = (0, 0)


class WasteType(IntEnum):
	GREEN = 1
	YELLOW = 2
	RED = 3


class ProblemType(IntEnum):
	ONLY_MOVE = 1
	MOVE_CATCH = 2
	PICK_ONE = 3
	ONLY_GREEN = 4
	GREEN_YELLOW = 5
	BALLS_ONLY = 6
	FULL = 7


class WasteState(object):
	_position: Tuple[int, int]
	_id: str
	_hold_state: int
	_holding_player: 'PlayerState'
	_points: float
	_time_penalty: float
	_identified: bool
	_was_picked: bool
	_waste_type: int
	
	def __init__(self, position: Tuple[int, int], obj_id: str, points: float = 1, time_penalty: float = 0.0, hold_state: int = HoldState.FREE.value,
	             waste_type: int = WasteType.GREEN, holding_player: 'PlayerState' = None, identified: bool = False):
		self._position = position
		self._id = obj_id
		self._hold_state = hold_state
		self._holding_player = holding_player
		self._points = points
		self._time_penalty = time_penalty
		self._identified = identified
		self._waste_type = waste_type
		self._was_picked = False

	@property
	def position(self) -> Tuple[int, int]:
		return self._position

	@property
	def id(self) -> str:
		return self._id

	@property
	def hold_state(self) -> int:
		return self._hold_state

	@property
	def holding_player(self) -> 'PlayerState':
		return self._holding_player

	@position.setter
	def position(self, new_pos: Tuple[int, int]) -> None:
		self._position = new_pos

	@hold_state.setter
	def hold_state(self, new_state: int) -> None:
		self._hold_state = new_state

	@holding_player.setter
	def holding_player(self, new_player: 'PlayerState') -> None:
		self._holding_player = new_player
	
	@property
	def identified(self) -> bool:
		return self._identified
	
	@property
	def waste_type(self) -> int:
		return self._waste_type
	
	@property
	def points(self) -> float:
		return self._points
	
	@property
	def time_penalty(self) -> float:
		return self._time_penalty
	
	@property
	def was_picked(self) -> bool:
		return self._was_picked
	
	@identified.setter
	def identified(self, new_val: bool) -> None:
		self._identified = new_val
	
	@was_picked.setter
	def was_picked(self, new_val: bool) -> None:
		self._was_picked = new_val
	
	@points.setter
	def points(self, new_val: float) -> None:
		self._points = new_val
	
	def deepcopy(self):
		new_obj = WasteState(self._position, self._id, self._points, self._time_penalty, identified=self._identified)
		new_obj.hold_state = self._hold_state
		return new_obj
	
	def __eq__(self, other):
		return isinstance(other, WasteState) and self._id == other._id and self._position == other._position
	
	def __repr__(self):
		return ("%s@(%d, %d), held_status: %s, identified? %r, picked? %r" %
				(self._id, self._position[0], self._position[1], HoldState(self._hold_state).name, self._identified, self._was_picked))
	
	def to_dict(self):
		return {"name": self._id, "position": self._position, "hold_state": self._hold_state, "identified": self._identified, "type": self._waste_type,
				"holding_player": self._holding_player.id if self._holding_player else None, "picked": self._was_picked}
	
	@classmethod
	def from_dict(cls, obj_dict):
		obj_dict = deepcopy(obj_dict)
		return WasteState(**obj_dict)


class PlayerState(object):
	_position: Tuple[int, int]
	_orientation: Tuple[int, int]
	_name: str
	_id: int
	_agent_type: int
	_held_object: List[WasteState]
	_reward: float
	_life_points: int

	def __init__(self, pos: Tuple[int, int], orientation: Tuple[int, int], agent_id: int, agent_name: str, agent_type: int, life_points: int = 20,
	             held_object: List[WasteState] = None):
		self._position = pos
		self._orientation = orientation
		self._agent_type = agent_type
		self._name = agent_name
		self._id = agent_id
		self._held_object = held_object
		self._reward = 0
		self._life_points = life_points

		if self._held_object is not None:
			for obj in self._held_object:
				assert isinstance(obj, WasteState)
				assert obj.position == self._position

	@property
	def position(self) -> Tuple[int, int]:
		return self._position

	@property
	def orientation(self) -> Tuple[int, int]:
		return self._orientation

	@property
	def agent_type(self) -> int:
		return self._agent_type

	@property
	def id(self) -> int:
		return self._id

	@property
	def name(self) -> str:
		return self._name

	@property
	def reward(self) -> float:
		return self._reward

	@property
	def held_objects(self) -> Optional[List[WasteState]]:
		if self._held_object is not None:
			return self._held_object.copy()
		else:
			return self._held_object

	@property
	def life_points(self) -> int:
		return self._life_points

	@position.setter
	def position(self, new_pos: Tuple[int, int]) -> None:
		self._position = new_pos

	@orientation.setter
	def orientation(self, new_orientation: Tuple[int, int]) -> None:
		self._orientation = new_orientation

	@reward.setter
	def reward(self, new_val: float) -> None:
		self._reward = new_val

	@life_points.setter
	def life_points(self, new_val: int) -> None:
		self._life_points = max(new_val, 0)

	def hold_object(self, other_obj: WasteState) -> None:
		assert isinstance(other_obj, WasteState), "[HOLD OBJECT ERROR] object is not an ObjectState"
		if self._held_object is not None:
			self._held_object.append(other_obj)
		else:
			self._held_object = [other_obj]

	def drop_object(self, obj_id: str) -> None:
		assert self.is_holding_object(), "[DROP OBJECT] holding no objects"

		for obj in self._held_object:
			if obj.id == obj_id:
				self._held_object.remove(obj)
				return

		print(colored('[DROP OBJECT] no object found with id %s' % obj_id, 'yellow'))

	def is_holding_object(self) -> bool:
		if self._held_object is not None:
			return len(self._held_object) > 0
		else:
			return False

	def deepcopy(self):
		held_objs = (None if self._held_object is None else [self._held_object[idx].deepcopy() for idx in range(len(self._held_object))])
		return PlayerState(self._position, self._orientation, self._id, self._name, self._agent_type, self._life_points, held_objs)

	def __eq__(self, other):
		return (
				isinstance(other, PlayerState)
				and self.position == other.position
				and self.orientation == other.orientation
				and self.held_objects == other.held_objects
				and self.life_points == other.life_points
		)

	def __hash__(self):
		return hash((self.position, self.orientation, self.held_objects))

	def __repr__(self):
		return "Agent {} at {} facing {} holding {} has {} hp.".format(self._name, self.position, self.orientation, str(self.held_objects), self.life_points)

	def to_dict(self):
		return {
				"name":        self._name,
				"position":    self.position,
				"orientation": self.orientation,
				"held_object": [self.held_objects[idx].to_dict() for idx in range(len(self._held_object))] if self.held_objects is not None else None,
				"life_points": self.life_points,
		}

	@staticmethod
	def from_dict(player_dict):
		player_dict = deepcopy(player_dict)
		held_obj = player_dict.get("held_object", None)
		if held_obj is not None:
			player_dict["held_object"] = [WasteState.from_dict(held_obj[idx]) for idx in range(len(held_obj))]
		return PlayerState(**player_dict)


# noinspection PyUnresolvedReferences
class ToxicWasteEnvV2(BaseToxicEnv):
	"""
	Collaborative game environment of toxic waste collection, useful for ad-hoc teamwork research.
	
	Version 2 - the agents have a fixed timelimit to collect all the waste and exist different types of waste that can have different impacts on the players'
	scoring and time remaining. Also, to help with identifying different wastes, the autonomous agent has access to an extra action of identification of waste.
	"""
	Observation = namedtuple("Observation",
							 ["field", "players", "objects", "game_finished", "game_timeout", "sight", "current_step", "time_left", "time_penalties",
							  "score"])

	_objects: List[WasteState]
	
	def __init__(self, terrain_size: Tuple[int, int], layout: str, max_players: int, max_objects: int, max_steps: int, rnd_seed: int, data_dir: Path,
	             require_facing: bool = False, agent_centered: bool = False, render_mode: List[str] = None, use_render: bool = False, slip: bool = False,
	             is_train: bool = False, random_init_pos: bool = False, dict_obs: bool = True, joint_obs: bool = False, pick_all: bool = False, problem_type: int = ProblemType.FULL):
		
		self._dict_obs = dict_obs
		self._is_train = is_train
		self._random_init_pos = random_init_pos
		self._slip = slip
		self._slip_prob = 0.0
		self._max_time = 0.0
		self._time_penalties = 0.0
		self._score = 0.0
		self._door_pos = (-1, 1)
		self._collect_all = pick_all or problem_type == ProblemType.BALLS_ONLY
		self._problem_type = problem_type
		super().__init__(terrain_size, layout, max_players, max_objects, max_steps, rnd_seed, 'v2', data_dir, require_facing, True, agent_centered,
		                 False, use_render, render_mode, joint_obs)
		if self._problem_type == ProblemType.ONLY_MOVE:
			self._reward_space = {'move': MOVE_PENALTY, 'deliver': 0.0, 'finish': ROOM_CLEAN, 'hold': 0.0, 'pick': PICK_REWARD, 'adjacent': 0.0, 'identify': 0.0}
		elif self._problem_type == ProblemType.MOVE_CATCH:
			self._reward_space = {'move': MOVE_PENALTY, 'deliver': 0.0, 'finish': ROOM_CLEAN, 'hold': 0.0, 'pick': 0.0, 'adjacent': 0.0, 'identify': 0.0}
		elif self._problem_type == ProblemType.PICK_ONE:
			self._reward_space = {'move': MOVE_PENALTY, 'deliver': DELIVER_WASTE, 'finish': ROOM_CLEAN, 'hold': 0.0, 'pick': PICK_REWARD, 'adjacent': 0.0, 'identify': 0.0}
		elif self._problem_type == ProblemType.FULL:
			self._reward_space = {'move': MOVE_PENALTY, 'deliver': DELIVER_WASTE, 'finish': ROOM_CLEAN, 'hold': HOLD_REWARD,
			                      'pick': PICK_REWARD, 'adjacent': ADJ_REWARD, 'identify': IDENTIFY_REWARD}
		else:
			self._reward_space = {'move': MOVE_PENALTY, 'deliver': DELIVER_WASTE, 'finish': ROOM_CLEAN, 'hold': HOLD_REWARD,
			                      'pick': PICK_REWARD, 'adjacent': ADJ_REWARD, 'identify': 0.0}
		self._start_time = time.time()
	
	###########################
	### GETTERS AND SETTERS ###
	###########################
	@property
	def slip(self) -> bool:
		return self._slip
	
	@slip.setter
	def slip(self, new_val: bool) -> None:
		self._slip = new_val
	
	@property
	def score(self) -> float:
		return self._score
	
	@property
	def door_pos(self) -> Tuple:
		return self._door_pos
	
	@property
	def has_pick_all(self) -> bool:
		return self._collect_all
	
	@property
	def problem_type(self) -> int:
		return self._problem_type
	
	def set_waste_color_pts(self, color: int, pts: float) -> None:
		for waste in self._objects:
			if waste.waste_type == color:
				waste.points = pts
	
	#######################
	### UTILITY METHODS ###
	#######################
	def add_object(self, position: Tuple[int, int], obj_id: str = 'ball', points: int = 1, time_penalty: float = 1, waste_type: int = WasteType.GREEN) -> bool:
		
		if self._n_objects < self._max_objects:
			is_identified = False if self._problem_type == ProblemType.FULL else True
			self._objects.append(WasteState(position, obj_id, points=points, time_penalty=time_penalty, identified=is_identified, waste_type=waste_type))
			self._n_objects += 1
			return True
		else:
			print(colored('[ADD_OBJECT] Max number of objects (%d) already reached, cannot add a new one.' % self._max_objects, 'yellow'))
			return False
	
	def _get_action_space(self) -> MultiDiscrete:
		return MultiDiscrete([len(Actions)] * self._n_players)
	
	def _get_observation_space(self) -> Union[gymnasium.spaces.Tuple, gymnasium.spaces.Dict]:
		
		# grid observation space
		if self._agent_centered_obs:
			grid_shape = (1 + 2 * self._agent_sight, 1 + 2 * self._agent_sight)
		
		else:
			grid_shape = (self._rows, self._cols)
		
		# agents layer: agent levels
		robots_min = np.zeros(grid_shape, dtype=np.int32)
		robots_max = np.ones(grid_shape, dtype=np.int32)
		humans_min = np.zeros(grid_shape, dtype=np.int32)
		humans_max = np.ones(grid_shape, dtype=np.int32)

		#orientation layer
		or_up_min = np.zeros(grid_shape, dtype=np.int32)
		or_up_max = np.ones(grid_shape, dtype=np.int32)
		or_down_min = np.zeros(grid_shape, dtype=np.int32)
		or_down_max = np.ones(grid_shape, dtype=np.int32)
		or_left_min = np.zeros(grid_shape, dtype=np.int32)
		or_left_max = np.ones(grid_shape, dtype=np.int32)
		or_right_min = np.zeros(grid_shape, dtype=np.int32)
		or_right_max = np.ones(grid_shape, dtype=np.int32)

		# waste layer: waste pos
		balls_min = np.zeros(grid_shape, dtype=np.int32)
		balls_max = np.ones(grid_shape, dtype=np.int32)
		green_min = np.zeros(grid_shape, dtype=np.int32)
		green_max = np.ones(grid_shape, dtype=np.int32)
		yellow_min = np.zeros(grid_shape, dtype=np.int32)
		yellow_max = np.ones(grid_shape, dtype=np.int32)
		red_min = np.zeros(grid_shape, dtype=np.int32)
		red_max = np.ones(grid_shape, dtype=np.int32)
		
		# access layer: i the cell available
		occupancy_min = np.zeros(grid_shape, dtype=np.int32)
		occupancy_max = np.ones(grid_shape, dtype=np.int32)
		
		# total layer
		min_obs = np.stack([robots_min, humans_min, or_up_min, or_down_min, or_left_min, or_right_min, balls_min, green_min, yellow_min, red_min, occupancy_min])
		max_obs = np.stack([robots_max, humans_max, or_up_max, or_down_max, or_left_max, or_right_max, balls_max, green_max, yellow_max, red_max, occupancy_max])
		
		if self._dict_obs:
			return gymnasium.spaces.Dict({'conv': Box(np.array(min_obs), np.array(max_obs), dtype=np.int32),
			                              'array': Box(np.array(0), np.array(self.max_steps), dtype=np.float32)})
		else:
			return gymnasium.spaces.Tuple([Box(np.array(min_obs), np.array(max_obs), dtype=np.int32),
			                               Box(np.array(0), np.array(self.max_steps), dtype=np.float32)])
	
	def setup_env(self) -> None:
		
		config_filepath = (Path(self._data_dir) if isinstance(self._data_dir, str) else self._data_dir) / 'configs' / 'layouts' / 'toxic_waste' / (self._room_layout + '.yaml')
		with open(config_filepath) as config_file:
			config_data = yaml.safe_load(config_file)
		field_data = config_data['field']
		players_data = config_data['players']
		objects_data = config_data['objects']
		self._slip_prob = float(config_data['slip_prob'])
		# self._max_time = float(config_data['max_train_time']) if self._is_train else float(config_data['max_game_time'])
		self._max_time = float(config_data['max_game_time'])
		self._max_time_steps = float(config_data['max_train_time'])
		config_sight = float(config_data['sight'])
		self._agent_sight = config_sight if config_sight > 0 else min(self._rows, self._cols)
		n_red = 0
		n_green = 0
		n_yellow = 0
		
		for row in range(self._rows):
			for col in range(self._cols):
				cell_val = field_data[row][col]
				if cell_val == ' ':
					pass
				elif cell_val == 'X':
					self._field[row, col] = CellEntity.COUNTER
				elif cell_val == 'T':
					self._field[row, col] = CellEntity.TOXIC
				elif cell_val == 'I':
					self._field[row, col] = CellEntity.ICE
				elif cell_val == 'D':
					self._field[row, col] = CellEntity.DOOR
					self._door_pos = (row, col)
				elif cell_val == 'G':
					'''points = 0.0 if self._problem_type == ProblemType.ONLY_MOVE else objects_data['green']['points']'''
					points = objects_data['green']['points']
					self.add_object((row, col), objects_data['green']['ids'][n_green], points,
					                objects_data['green']['time_penalty'], waste_type=WasteType.GREEN)
					self._field[row, col] = CellEntity.COUNTER
					n_green += 1
				elif cell_val == 'R':
					points = 0.0 if not (self._problem_type == ProblemType.FULL or self._problem_type == ProblemType.BALLS_ONLY) else objects_data['red']['points']
					self.add_object((row, col), objects_data['red']['ids'][n_red], points,
					                objects_data['red']['time_penalty'], waste_type=WasteType.RED)
					self._field[row, col] = CellEntity.COUNTER
					n_red += 1
				elif cell_val == 'Y':
					'''points = 0.0 if (self._problem_type == ProblemType.ONLY_MOVE or self._problem_type == ProblemType.ONLY_GREEN) else objects_data['yellow']['points']'''
					points = objects_data['yellow']['points']
					self.add_object((row, col), objects_data['yellow']['ids'][n_yellow], points,
					                objects_data['yellow']['time_penalty'], waste_type=WasteType.YELLOW)
					self._field[row, col] = CellEntity.COUNTER
					n_yellow += 1
				elif cell_val.isdigit():
					for player in players_data:
						if str(player['id']) == cell_val: nxt_player_data = player
					# nxt_player_data = players_data[self._n_players]
					# noinspection PyTypeChecker
					self.add_player((row, col), tuple(nxt_player_data['orientation']), nxt_player_data['id'], nxt_player_data['name'],
					                AgentType[nxt_player_data['type'].upper()].value)
				else:
					print(colored("[SETUP_ENV] Cell value %s not recognized, considering empty cell" % cell_val, 'yellow'))
					continue
	
	def is_game_finished(self) -> bool:
		player_at_door = any([self._field[p.position[0], p.position[1]] == CellEntity.DOOR for p in self.players if p.agent_type == AgentType.HUMAN])
		if self._collect_all:
			remain_balls = [obj for obj in self.objects if obj.hold_state != HoldState.DISPOSED]
			if self._problem_type == ProblemType.ONLY_GREEN:
				return player_at_door and all([(ball.waste_type == WasteType.RED or ball.waste_type == WasteType.YELLOW) for ball in remain_balls])
			elif self._problem_type == ProblemType.BALLS_ONLY: # catch all balls
				return player_at_door and not remain_balls
			else:
				return player_at_door and all([ball.waste_type == WasteType.RED for ball in remain_balls])
		else:
			if self._problem_type == ProblemType.MOVE_CATCH:
				return player_at_door and any([ball.was_picked for ball in self.objects])
			elif self._problem_type == ProblemType.PICK_ONE:
				return player_at_door and any([ball.hold_state == HoldState.DISPOSED for ball in self.objects])
			else:
				return player_at_door
	
	def is_game_timedout(self) -> bool:
		return self.get_time_left() <= 0 if not self._is_train else self._current_step >= self.max_steps
	
	def move_ice(self, move_agent: PlayerState, next_position: Tuple) -> Tuple:
		
		agent_pos = move_agent.position
		right_move = (next_position[0] - agent_pos[0], next_position[1] - agent_pos[1])
		wrong_moves = [direction.value for direction in ActionDirection if direction.value != right_move and direction.value != (0, 0)]
		n_wrong_moves = len(wrong_moves)
		moves_prob = np.array([1 - self._slip_prob] + [self._slip_prob / n_wrong_moves] * n_wrong_moves)
		possible_positions = ([next_position] + [(max(min(wrong_move[0] + agent_pos[0], self._rows), 0), max(min(wrong_move[1] + agent_pos[1], self.cols), 0))
		                                         for wrong_move in wrong_moves])
		return possible_positions[self._np_random.choice(range(len(possible_positions)), p=moves_prob)]
	
	def get_time_left(self) -> float:
		
		curr_time = self._max_time - (time.time() - self._start_time)
		return curr_time - self._time_penalties
	
	def get_object_facing(self, player: PlayerState) -> Optional[WasteState]:
		facing_pos = (player.position[0] + player.orientation[0], player.position[1] + player.orientation[1])
		for obj in self._objects:
			if obj.position == facing_pos and obj.hold_state == HoldState.FREE:
				return obj
		
		return None
	
	####################
	### MAIN METHODS ###
	####################
	def create_observation(self) -> Observation:
		return self.Observation(field=self.field,
		                        players=self.players,
		                        objects=self.objects,
		                        game_finished=self.is_game_finished(),
		                        game_timeout=self.is_game_timedout(),
		                        sight=self._agent_sight,
		                        current_step=self._current_step,
		                        time_left=self.get_time_left(),
		                        time_penalties=self._time_penalties,
		                        score=self._score)
	
	def get_env_log(self) -> str:
		
		env_log = 'Environment state:\nPlayer states:\n'
		for player in self._players:
			env_log += '\t- ' + str(player) + '\n'
		
		env_log += 'Object states:\n'
		for obj in self._objects:
			if obj.hold_state != HoldState.HELD:
				env_log += '\t- ' + str(obj) + '\n'
		
		env_log += 'Current timestep: %d\nGame is finished: %r\n' % (self._current_step, self.is_game_finished())
		env_log += 'Game has timed out: %r\nTime left: %f' % (self.is_game_timedout(), self.get_time_left())
		
		return env_log
	
	def get_full_env_log(self) -> str:
		
		env_log = 'Environment state:\nPlayer states:\n'
		for player in self._players:
			env_log += '\t- ' + str(player) + '\n'
		
		env_log += 'Object states:\n'
		for obj in self._objects:
			if obj.hold_state != HoldState.HELD:
				env_log += '\t- ' + str(obj) + '\n'
		
		field = self._field.copy()
		for player in self._players:
			field[player.position[0], player.position[1]] = CellEntity.AGENT
		env_log += 'Field layout: %s\n' % str(field)
		env_log += 'Current timestep: %d\nGame is finished: %r\n' % (self._current_step, self.is_game_finished())
		env_log += 'Game has timed out: %r\nTime left: %f' % (self.is_game_timedout(), self.get_time_left())
		
		return env_log
	
	def make_obs_grid(self) -> Union[np.ndarray, List]:
		
		if self._agent_centered_obs:
			layers_size = (self._rows + 2 * self._agent_sight, self._cols + 2 * self._agent_sight)
			robot_layer = np.zeros(layers_size, dtype=np.int32)
			human_layer = np.zeros(layers_size, dtype=np.int32)
			balls_layer = np.zeros(layers_size, dtype=np.int32)
			green_layer = np.zeros(layers_size, dtype=np.int32)
			yellow_layer = np.zeros(layers_size, dtype=np.int32)
			red_layer = np.zeros(layers_size, dtype=np.int32)
			up_layer = np.zeros(layers_size, dtype=np.int32)
			down_layer = np.zeros(layers_size, dtype=np.int32)
			left_layer = np.zeros(layers_size, dtype=np.int32)
			right_layer = np.zeros(layers_size, dtype=np.int32)
			occupancy_layer = np.ones(layers_size, dtype=np.int32)
			occupancy_layer[:self._agent_sight, :] = 0
			occupancy_layer[-self._agent_sight:, :] = 0
			occupancy_layer[:, :self._agent_sight] = 0
			occupancy_layer[:, -self._agent_sight:] = 0
			
			for agent in self._players:
				pos = agent.position
				orient = agent.orientation
				# position
				if agent.agent_type == AgentType.HUMAN:
					human_layer[pos[0] + self._agent_sight, pos[1] + self._agent_sight] = 1
				else:
					robot_layer[pos[0] + self._agent_sight, pos[1] + self._agent_sight] = 1
				occupancy_layer[pos[0] + self._agent_sight, pos[1] + self._agent_sight] = 0
				
				#orientation
				if agent.orientation == (-1,0): up_layer[pos[0] + self._agent_sight, pos[1] + self._agent_sight] = 1
				elif agent.orientation == (1,0): down_layer[pos[0] + self._agent_sight, pos[1] + self._agent_sight] = 1
				elif agent.orientation == (0,-1): left_layer[pos[0] + self._agent_sight, pos[1] + self._agent_sight] = 1
				elif agent.orientation == (0,1): right_layer[pos[0] + self._agent_sight, pos[1] + self._agent_sight] = 1
			 
			for obj in self._objects:
				pos = obj.position
				balls_layer[pos[0] + self._agent_sight, pos[1] + self._agent_sight] = 1
				occupancy_layer[pos[0] + self._agent_sight, pos[1] + self._agent_sight] = 0
				if obj.identified:
					if obj.waste_type == WasteType.GREEN:
						green_layer[pos[0] + self._agent_sight, pos[1] + self._agent_sight] = 1
					elif obj.waste_type == WasteType.YELLOW:
						yellow_layer[pos[0] + self._agent_sight, pos[1] + self._agent_sight] = 1
					elif obj.waste_type == WasteType.RED:
						red_layer[pos[0] + self._agent_sight, pos[1] + self._agent_sight] = 1
			
			for row in range(self._rows):
				for col in range(self._cols):
					if self._field[row, col] == CellEntity.COUNTER:
						occupancy_layer[row + self._agent_sight, col + self._agent_sight] = 0
			
			obs = np.stack([robot_layer, human_layer, up_layer, down_layer, left_layer, right_layer, balls_layer, green_layer, yellow_layer, red_layer, occupancy_layer])
			padding = 2 * self._agent_sight + 1
			time_left = self.get_time_left()
			
			if self._dict_obs:
				return [{'conv': obs[:, a.position[0]:a.position[0] + padding, a.position[1]:a.position[1] + padding], 'array': np.array(time_left)}
				        for a in self._players]
			else:
				return np.array([np.array([obs[:, a.position[0]:a.position[0] + padding, a.position[1]:a.position[1] + padding], np.array(time_left)],
				                          dtype=object)
				                 for a in self._players])
		
		else:
			layers_size = (self._rows, self._cols)
			robot_layer = np.zeros(layers_size, dtype=np.int32)
			human_layer = np.zeros(layers_size, dtype=np.int32)
			balls_layer = np.zeros(layers_size, dtype=np.int32)
			green_layer = np.zeros(layers_size, dtype=np.int32)
			yellow_layer = np.zeros(layers_size, dtype=np.int32)
			red_layer = np.zeros(layers_size, dtype=np.int32)
			occupancy_layer = np.ones(layers_size, dtype=np.int32)
			acting_layer = np.zeros((self._n_players, *layers_size), dtype=np.int32)
			
			for agent_idx in range(self._n_players):
				pos = self._players[agent_idx].position
				if self._players[agent_idx].agent_type == AgentType.HUMAN:
					human_layer[pos[0], pos[1]] = 1
				else:
					robot_layer[pos[0], pos[1]] = 1
				occupancy_layer[pos[0], pos[1]] = 0
				acting_layer[agent_idx, pos[0], pos[1]] = 1
			
			for obj in self._objects:
				pos = obj.position
				balls_layer[pos[0], pos[1]] = 1
				occupancy_layer[pos[0], pos[1]] = 0
				if obj.identified:
					if obj.waste_type == WasteType.GREEN:
						green_layer[pos[0], pos[1]] = 1
					elif obj.waste_type == WasteType.YELLOW:
						yellow_layer[pos[0], pos[1]] = 1
					elif obj.waste_type == WasteType.RED:
						red_layer[pos[0], pos[1]] = 1
			
			for row in range(self._rows):
				for col in range(self._cols):
					if self._field[row, col] == CellEntity.COUNTER:
						occupancy_layer[row, col] = 0
			time_left = self.get_time_left()
			
			if self._dict_obs:
				if self._joint_obs:
					return [{'conv': np.stack([robot_layer, human_layer, balls_layer, green_layer, yellow_layer, red_layer, occupancy_layer]), 'array': np.array(time_left)}]
				else:
					return [{'conv': np.stack([robot_layer, human_layer, balls_layer, green_layer, yellow_layer, red_layer, occupancy_layer, acting_layer[idx]]),
					         'array': np.array(time_left)}
					        for idx in range(self._n_players)]
			else:
				if self._joint_obs:
					return np.array([np.stack([robot_layer, human_layer, green_layer, yellow_layer, red_layer, occupancy_layer]), np.array(time_left)], dtype=object)
				else:
					return np.array([np.array([np.stack([robot_layer, green_layer, yellow_layer, red_layer, occupancy_layer, acting_layer[idx]]),
					                           np.array(time_left)],
					                          dtype=object)
					                 for idx in range(self._n_players)])
	
	def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
		
		self._max_time = 0.0
		self._time_penalties = 0.0
		self._score = 0.0
		self._door_pos = (-1, 1)
		obs, info = super().reset(seed=seed, options=options)

		if self._is_train or self._random_init_pos:
			valid_pos = list(product(range(self.rows), range(self.cols)))
			for pos in np.transpose(np.nonzero(self._field)):
				valid_pos.remove(tuple(pos))

			for p in self.players:
				row, col = self._np_random.choice(valid_pos)
				p.position = tuple([row, col])
				valid_pos.remove(tuple([row, col]))
		
		self._start_time = time.time()
		
		return obs, info
	
	def step(self, actions: List[int]) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
		
		slip_agents, agent_bonus = self.execute_transitions(actions)
		finished = self.is_game_finished()
		'''if self._problem_type == ProblemType.ONLY_MOVE:
			rewards = np.zeros(self.n_players) if finished else MOVE_PENALTY * np.ones(self.n_players)
		else:
			rewards = np.array([player.reward for player in self._players])
		# rewards = np.array([self._score + agent_bonus[idx] for idx in range(self.n_players)])'''
		rewards = np.array([player.reward for player in self._players])
		timeout = self.is_game_timedout()
		return self.make_obs(), rewards, finished, timeout, {'agents_slipped': slip_agents}
	
	def execute_transitions(self, actions: List[int]) -> Tuple[List[int], List[float]]:
		
		self._current_step += 1
		old_positions = []
		bonus_pts = [0] * self.n_players
		for player in self._players:
			player.reward = self._reward_space['move']
			old_positions += [player.position]
		
		# Process action list
		new_positions = []
		slip_agents = []
		agents_disposed_waste = []
		waste_disposed = {}
		for agent_idx in range(self.n_players):
			act = actions[agent_idx]
			acting_player = self._players[agent_idx]
			act_direction = ActionDirection[Actions(act).name].value
			if act != Actions.INTERACT and act != Actions.STAY:# and act != Actions.IDENTIFY:
				acting_player.orientation = act_direction
			next_pos = (max(min(acting_player.position[0] + act_direction[0], self._rows), 0),
			            max(min(acting_player.position[1] + act_direction[1], self.cols), 0))
			if not (self._field[next_pos] == CellEntity.DOOR or self._field[next_pos] == CellEntity.EMPTY or
			        self._field[next_pos] == CellEntity.TOXIC or self._field[next_pos] == CellEntity.ICE):
				new_positions.append(acting_player.position)
			elif self._slip and self._field[acting_player.position] == CellEntity.ICE:
				new_positions.append(self.move_ice(acting_player, next_pos))
				slip_agents.append(acting_player.id)
			else:
				new_positions.append(next_pos)
			act = actions[agent_idx]
			acting_player = self._players[agent_idx]
			act_direction = ActionDirection[Actions(act).name].value

		for agent_idx in range(self.n_players):
			act = actions[agent_idx]
			acting_player = self._players[agent_idx]
			act_direction = ActionDirection[Actions(act).name].value
			# Handle INTERACT action is only necessary for human agents
			if act == Actions.INTERACT and acting_player.agent_type == AgentType.HUMAN:
				facing_pos = (acting_player.position[0] + acting_player.orientation[0], acting_player.position[1] + acting_player.orientation[1])
				adjacent_agent = self.get_agent_facing(acting_player)
				if acting_player.is_holding_object():
					if adjacent_agent is not None:  # facing an agent
						adj_agent_idx = self._players.index(adjacent_agent)
						adj_agent_action = actions[adj_agent_idx]
						adj_agent_type = adjacent_agent.agent_type
						# check if the agent is a robot and is not trying to move
						if adj_agent_type == AgentType.ROBOT and adj_agent_action == Actions.STAY:#and adjacent_agent.position == new_positions[adj_agent_idx]:
							# can only place trash if agent and robot are looking at each other
							if self.require_facing and not self.are_facing(acting_player, adjacent_agent):
								continue
							# Place object in robot
							place_obj = acting_player.held_objects[0]
							acting_player.drop_object(place_obj.id)
							place_obj.hold_state = HoldState.DISPOSED
							place_obj.holding_player = adjacent_agent
							place_obj.position = (-1, -1)
							adjacent_agent.hold_object(place_obj)
							agents_disposed_waste.append(acting_player)
							agents_disposed_waste.append(adjacent_agent)
							# waste_disposed[acting_player.id] = place_obj.points
							waste_disposed[acting_player.id] = self._reward_space['deliver'] + place_obj.points
							waste_disposed[adjacent_agent.id] = self._reward_space['deliver'] + place_obj.points
							self._score += place_obj.points
					else:
						# Drop object to the field
						dropped_obj = acting_player.held_objects[0]
						if dropped_obj.hold_state == HoldState.HELD and self.free_pos(facing_pos):
							acting_player.drop_object(dropped_obj.id)
							dropped_obj.position = facing_pos
							dropped_obj.hold_state = HoldState.FREE
							dropped_obj.holding_player = None
				
				else:
					if adjacent_agent is None:
						# Pick object from counter or floor
						for obj in self._objects:
							if obj.position == facing_pos and obj.hold_state == HoldState.FREE:								
								pick_obj = obj
								pick_obj.position = acting_player.position
								pick_obj.hold_state = HoldState.HELD
								pick_obj.holding_player = acting_player
								pick_obj.identified = True
								acting_player.hold_object(pick_obj)
								if not pick_obj.was_picked:
									acting_player.reward += self._reward_space['pick']
									bonus_pts[agent_idx] += self._reward_space['pick']
									pick_obj.was_picked = True	
						# self._time_penalties += pick_obj.time_penalty			# Uncomment if it is supposed to apply penalty at pickup
			
			# IDENTIFY action only has impact by robot agents
			elif act == Actions.INTERACT and acting_player.agent_type == AgentType.ROBOT:
				object_facing = self.get_object_facing(acting_player)
				if object_facing is not None and not object_facing.identified:
					acting_player.reward = self._reward_space['identify']
					bonus_pts[agent_idx] += self._reward_space['identify']
					object_facing.identified = True
		
		# Handle movement and collisions
		# Check for collisions (more than one agent moving to same position or two agents switching position)
		can_move = []
		for idx in range(self._n_players):
			curr_pos = old_positions[idx]
			next_pos = new_positions[idx]
			add_move = True
			for idx2 in range(self._n_players):
				if idx2 == idx:
					continue
				if ((next_pos == old_positions[idx2] and old_positions[idx2] == new_positions[idx2]) or new_positions[idx2] == next_pos or
						(curr_pos == new_positions[idx2] and next_pos == old_positions[idx2])):
					add_move = False
					break
			if add_move:
				can_move.append(idx)
		
		# Update position for agents with valid movements
		for idx in can_move:
			moving_player = self._players[idx]
			old_pos = old_positions[idx]
			next_pos = new_positions[idx]
			ball_pos = [obj.position for obj in self.objects]
			if idx == AgentType.HUMAN and moving_player.is_holding_object():
				self._time_penalties += sum([obj.time_penalty for obj in moving_player.held_objects])	# When the agent moves holding waste apply time penalty
				self._players[AgentType.ROBOT].reward = self._reward_space['hold']
			if old_pos != next_pos and next_pos not in ball_pos:
				moving_player.position = next_pos
				if moving_player.is_holding_object():
					for obj in moving_player.held_objects:
						if obj.hold_state != HoldState.DISPOSED:
							obj.position = next_pos
		
		if self.is_game_finished():
			# Game finished reward
			time_left = self.get_time_left() if not self._is_train else (self.max_steps - self._current_step)
			max_time = self._max_time if not self._is_train else self.max_steps
			#self._score += (time_left / max_time) * self._score
			for player in self._players:
				#player.reward += self._reward_space['finish'] * self._score
				player.reward += self._reward_space['finish'] + self._score
				
		else:
			for player in self._players:
				if player in agents_disposed_waste:
					# Disposal reward
					player.reward += waste_disposed[player.id]
				else:
					# Adjacency reward
					facing_agent = self.get_agent_facing(player)
					if (facing_agent is not None and
							((player.agent_type == AgentType.HUMAN and facing_agent.agent_type == AgentType.ROBOT and player.is_holding_object()) or
							 (player.agent_type == AgentType.ROBOT and facing_agent.agent_type == AgentType.HUMAN and facing_agent.is_holding_object()))):
						# if players have to face each other, reward only given when they are facing
						if (self.require_facing and self.are_facing(player, facing_agent)) or not self.require_facing:
							player.reward += self._reward_space['adjacent']
							bonus_pts[self._players.index(player)] += self._reward_space['adjacent']
		
		return slip_agents, bonus_pts
