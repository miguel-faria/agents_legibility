#! /usr/bin/env python

import numpy as np
import gymnasium

from enum import IntEnum, Enum
from gymnasium.utils import seeding
from gymnasium.spaces import Box, MultiDiscrete
from gymnasium import Env
from typing import List, Tuple, Any, Dict, Optional
from copy import deepcopy
from termcolor import colored
from collections import namedtuple
from pathlib import Path


MOVE_REWARD = 0.0
HOLD_REWARD = -3.0
DELIVER_WASTE = 10
ROOM_CLEAN = 50


class AgentType(IntEnum):
	HUMAN = 0
	ROBOT = 1


class HoldState(IntEnum):
	FREE = 0
	HELD = 1
	DISPOSED = 2


class CellEntity(IntEnum):
	EMPTY = 0
	COUNTER = 1
	TOXIC = 2
	ICE = 3
	AGENT = 4
	OBJECT = 5
	DOOR = 6


class Actions(IntEnum):
	UP = 0
	DOWN = 1
	LEFT = 2
	RIGHT = 3
	INTERACT = 4
	STAY = 5


class ActionDirection(Enum):
	UP = (-1, 0)
	DOWN = (1, 0)
	LEFT = (0, -1)
	RIGHT = (0, 1)
	INTERACT = (0, 0)
	STAY = (0, 0)


class WasteState(object):
	_position: Tuple[int, int]
	_id: str
	_hold_state: int
	_holding_player: 'PlayerState'
	
	def __init__(self, position: Tuple[int, int], obj_id: str, hold_state: int = HoldState.FREE.value, holding_player: 'PlayerState' = None):
		self._position = position
		self._id = obj_id
		self._hold_state = hold_state
		self._holding_player = holding_player
	
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
	
	def deepcopy(self):
		new_obj = WasteState(self._position, self._id)
		new_obj.hold_state = self._hold_state
		return new_obj
	
	def __eq__(self, other):
		return isinstance(other, WasteState) and self._id == other._id and self._position == other._position
	
	def __hash__(self):
		return hash((self._id, self._position))
	
	def __repr__(self):
		return "%s@(%d, %d), held_status: %s" % (self._id, self._position[0], self._position[1], HoldState(self._hold_state).name)
	
	def to_dict(self):
		return {"name": self._id, "position": self._position, "hold_state": self._hold_state,
				"holding_player": self._holding_player.id if self._holding_player else None}
	
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
	
	def __init__(self, pos: Tuple[int, int], orientation: Tuple[int, int], agent_id: int, agent_name: str, agent_type: int,
				 held_object: List[WasteState] = None):
		self._position = pos
		self._orientation = orientation
		self._agent_type = agent_type
		self._name = agent_name
		self._id = agent_id
		self._held_object = held_object
		self._reward = 0
		
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
	
	@position.setter
	def position(self, new_pos: Tuple[int, int]) -> None:
		self._position = new_pos
	
	@orientation.setter
	def orientation(self, new_orientation: Tuple[int, int]) -> None:
		self._orientation = new_orientation
	
	@reward.setter
	def reward(self, new_val: float) -> None:
		self._reward = new_val
	
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
		return PlayerState(self._position, self._orientation, self._id, self._name, self._agent_type, held_objs)
	
	def __eq__(self, other):
		return (
				isinstance(other, PlayerState)
				and self.position == other.position
				and self.orientation == other.orientation
				and self.held_objects == other.held_objects
		)
	
	def __hash__(self):
		return hash((self.position, self.orientation, self.held_objects))
	
	def __repr__(self):
		return "Agent {} at {} facing {} holding {}".format(self._name, self.position, self.orientation, str(self.held_objects))
	
	def to_dict(self):
		return {
			"name": self._name,
			"position": self.position,
			"orientation": self.orientation,
			"held_object": [self.held_objects[idx].to_dict() for idx in range(len(self._held_object))] if self.held_objects is not None else None,
		}
	
	@staticmethod
	def from_dict(player_dict):
		player_dict = deepcopy(player_dict)
		held_obj = player_dict.get("held_object", None)
		if held_obj is not None:
			player_dict["held_object"] = [WasteState.from_dict(held_obj[idx]) for idx in range(len(held_obj))]
		return PlayerState(**player_dict)


# noinspection PyUnresolvedReferences
class BaseToxicEnv(Env):
	Observation = namedtuple("Observation", ["field", "players", "objects", "game_finished", "game_timeout", "sight", "current_step"])
	
	action_space: MultiDiscrete
	observation_space: gymnasium.spaces.Tuple
	_reward_space: Dict[str, float]
	_objects: List[WasteState]
	_players: List[PlayerState]
	
	def __init__(self, terrain_size: Tuple[int, int], layout: str, max_players: int, max_objects: int, max_steps: int, rnd_seed: int, env_id: str, data_dir: Path,
				 require_facing: bool = False, layer_obs: bool = False, agent_centered: bool = False, use_encoding: bool = False, use_render: bool = False,
				 render_mode: List[str] = None, joint_obs: bool = False):
		
		self._np_random, _ = seeding.np_random(rnd_seed)
		self._env_id = env_id
		self._rows, self._cols = terrain_size
		self._players: List[PlayerState] = []
		self._objects: List[WasteState] = []
		self._field: np.ndarray = np.zeros((self._rows, self._cols))
		self._max_players = max_players
		self._max_objects = max_objects
		self._n_players = 0
		self._n_objects = 0
		self._room_layout = layout
		self._max_time_steps = max_steps
		self._current_step = 0
		self._n_actions = len(Actions)
		self._agent_sight = 0
		self._require_facing = require_facing
		self._render = None
		self._use_render = use_render
		self._use_layer_obs = layer_obs
		self._joint_obs = joint_obs
		self._agent_centered_obs = agent_centered
		self._use_encoding = use_encoding
		self._data_dir = data_dir
		self._reward_space = {}
		self.setup_env()
		
		self.action_space = self._get_action_space()
		self.observation_space = gymnasium.spaces.Tuple([self._get_observation_space()] * self._n_players)
		self.seed(rnd_seed)
		if render_mode is None:
			self.metadata = {"render_modes": ['human']}
			self._show_viewer = True
			self.render_mode = 'human'
		else:
			self.metadata = {"render_modes": render_mode}
			self._show_viewer = 'human' in render_mode
			self.render_mode = 'rgb_array' if 'rgb_array' in render_mode else 'human'
	
	###########################
	### GETTERS AND SETTERS ###
	###########################
	@property
	def env_id(self) -> str:
		return self._env_id
	
	@property
	def rows(self) -> int:
		return self._rows
	
	@property
	def cols(self) -> int:
		return self._cols
	
	@property
	def players(self) -> List[PlayerState]:
		return self._players
	
	@property
	def objects(self) -> List[WasteState]:
		return self._objects
	
	@property
	def max_players(self) -> int:
		return self._max_players
	
	@property
	def max_objects(self) -> int:
		return self._max_objects
	
	@property
	def n_players(self) -> int:
		return self._n_players
	
	@property
	def n_objects(self) -> int:
		return self._n_objects
	
	@property
	def layout(self) -> str:
		return self._room_layout
	
	@property
	def max_steps(self) -> int:
		return self._max_time_steps
	
	@property
	def require_facing(self) -> bool:
		return self._require_facing
	
	@property
	def field(self) -> np.ndarray:
		return self._field
	
	@property
	def use_render(self) -> bool:
		return self._use_render
	
	@property
	def agent_centered(self) -> bool:
		return self._agent_centered_obs
	
	@property
	def use_joint_obs(self) -> bool:
		return self._joint_obs
	
	@property
	def reward_space(self) -> Dict[str, float]:
		return self._reward_space
	
	@layout.setter
	def layout(self, new_layout: str) -> None:
		self._room_layout = new_layout
	
	@use_render.setter
	def use_render(self, new_val: bool) -> None:
		self._use_render = new_val
	
	@reward_space.setter
	def reward_space(self, new_space: Dict[str, float]) -> None:
		self._reward_space = new_space
	
	def seed(self, seed=None):
		self._np_random, seed = seeding.np_random(seed)
		self.action_space.seed(seed)
		if isinstance(self.action_space, gymnasium.spaces.Tuple):
			for idx in range(len(self.action_space)):
				self.action_space[idx].seed(seed)
		self.observation_space.seed(seed)
		if isinstance(self.observation_space, gymnasium.spaces.Tuple):
			for idx in range(len(self.observation_space)):
				self.observation_space[idx].seed(seed)
		return [seed]
	
	def set_reward_value(self, reward_type: str, reward_value: float) -> None:
		self._reward_space[reward_type] = reward_value
	
	#######################
	### UTILITY METHODS ###
	#######################
	def _get_observation_space(self) -> Box:
		raise NotImplementedError('The observation space has to be implemented by the version of the environment to use')
	
	def _get_action_space(self) -> MultiDiscrete:
		raise NotImplementedError('The action space has to be implemented by the version of the environment to use')
	
	def get_filled_field(self) -> np.ndarray:
		
		field = self._field.copy()
		for agent in self._players:
			field[agent.position] = CellEntity.AGENT
		
		for obj in self._objects:
			field[obj.position] = CellEntity.OBJECT
		
		return field
	
	def setup_env(self) -> None:
		raise NotImplementedError('The environment setup has to be specified in each version of the environment')
	
	@staticmethod
	def encode_orientation(orientation: Tuple) -> Tuple[int, int]:
		raise NotImplementedError('The orientation of the actions is dependant of the actions in each version of the environment')
	
	@staticmethod
	def get_adj_pos(position: Tuple, ignore_diag: bool = True) -> List[Tuple[int, int]]:
		if ignore_diag:
			return [(position[0] - 1, position[1]), (position[0] + 1, position[1]), (position[0], position[1] - 1), (position[0], position[1] - 1)]
		
		else:
			return [(position[0] - 1, position[1]), (position[0] + 1, position[1]), (position[0], position[1] - 1),
					(position[0], position[1] + 1), (position[0] - 1, position[1] - 1), (position[0] + 1, position[1] - 1),
					(position[0] - 1, position[1] + 1), (position[0] + 1, position[1] + 1)]
	
	def disposed_objects(self) -> List[WasteState]:
		return [obj for obj in self._objects if obj.hold_state == HoldState.DISPOSED]
	
	def is_game_timedout(self) -> bool:
		return self._current_step >= self.max_steps
	
	def is_over(self) -> bool:
		return self.is_game_finished() or self.is_game_timedout()
	
	def is_game_finished(self) -> bool:
		raise NotImplementedError('The end state condition is specific of each environment\'s version')
	
	@staticmethod
	def are_facing(agent_1: PlayerState, agent_2: PlayerState) -> bool:
		return (agent_1.orientation[0] + agent_2.orientation[0]) == 0 and (agent_1.orientation[1] + agent_2.orientation[1]) == 0
	
	def free_pos(self, pos: Tuple[int, int]) -> bool:
		for player in self._players:
			if player.position == pos:
				return False
		
		for obj in self._objects:
			if obj.position == pos:
				return False
		
		return True
	
	def get_agent_facing(self, player: PlayerState) -> Optional[PlayerState]:
		facing_pos = (player.position[0] + player.orientation[0], player.position[1] + player.orientation[1])
		for agent in self._players:
			if agent.position == facing_pos:
				return agent
		
		return None
	
	####################
	### MAIN METHODS ###
	####################
	def get_env_log(self) -> str:
		raise NotImplementedError('The environment log is specific for each version')

	def get_full_env_log(self) -> str:
		raise NotImplementedError('The environment log is specific for each version')
	
	def create_observation(self) -> Observation:
		return self.Observation(field=self.field,
								players=self.players,
								objects=self.objects,
								game_finished=self.is_game_finished(),
								game_timeout=self.is_game_timedout(),
								sight=self._agent_sight,
								current_step=self._current_step)
	
	def add_player(self, position: Tuple[int, int], orientation: Tuple[int, int] = (-1, 0), agent_id: int = 0, agent_name: str = 'human', agent_type: int = AgentType.HUMAN,
				   held_objs: List[WasteState] = None) -> bool:
		
		if self._n_players < self._max_players:
			self._players.append(PlayerState(position, orientation, agent_id, agent_name, agent_type, held_objs))
			self._n_players += 1
			self._players.sort(key=lambda p: p._id)
			return True
		else:
			print(colored('[ADD_PLAYER] Max number players (%d) already reached, cannot add a new one.' % self._max_players, 'yellow'))
			return False
	
	def add_object(self, position: Tuple[int, int], obj_id: str = 'ball') -> bool:
		
		if self._n_objects < self._max_objects:
			self._objects.append(WasteState(position, obj_id))
			self._n_objects += 1
			return True
		else:
			print(colored('[ADD_OBJECT] Max number of objects (%d) already reached, cannot add a new one.' % self._max_objects, 'yellow'))
			return False
	
	def remove_player(self, player_id: int) -> bool:
		
		for player in self._players:
			if player.id == player_id:
				if player.is_holding_object():
					if len(player.held_objects) > 1:
						obj_idx = 0
						adj_pos = self.get_adj_pos(player.position, False)
						n_adj_pos = len(adj_pos)
						for obj in player.held_objects:
							obj.hold_state = HoldState.FREE
							if obj_idx < n_adj_pos:
								obj.position = adj_pos[obj_idx]
							else:
								obj.position = player.position
							obj_idx += 1
					else:
						obj = player.held_objects[0]
						obj.hold_state = HoldState.FREE.value
				self._players.remove(player)
				self._n_players -= 1
				return True
		
		print(colored('Player with id %s not found in list' % player_id, 'yellow'))
		return False
	
	def remove_object(self, obj_id: str) -> bool:
		
		for obj in self._objects:
			if obj.id == obj_id:
				if obj.hold_state == HoldState.HELD or obj.hold_state == HoldState.DISPOSED:
					obj.holding_player.held_objects.remove(obj)
				self._objects.remove(obj)
				self._n_objects -= 1
				return True
		
		print(colored('Object with id %s not found in list' % obj_id, 'yellow'))
		return False
	
	def make_obs(self) -> np.ndarray:
		if self._use_layer_obs:
			return self.make_obs_grid()
		elif self._use_encoding:
			return self.make_obs_dqn()
		else:
			return self.make_obs_array()
	
	def make_obs_array(self) -> np.ndarray:
		raise NotImplementedError('The observation array is specific for each environment version')
	
	def make_obs_grid(self) -> np.ndarray:
		raise NotImplementedError('The observation grid is specific for each environment version')
	
	def make_obs_dqn(self) -> np.ndarray:
		raise NotImplementedError('The DQN observation array is specific for each environment version')
	
	def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
		
		if seed is not None:
			self.seed(seed)
		
		self._current_step = 0
		self._n_players = 0
		self._n_objects = 0
		self._players: List[PlayerState] = []
		self._objects: List[WasteState] = []
		self._field: np.ndarray = np.zeros((self._rows, self._cols))
		self.setup_env()
		
		return self.make_obs(), {}
	
	def execute_transitions(self, actions: List[int]):
		raise NotImplementedError('Transitions are dependent on the environment version\'s dynamics.')
	
	def step(self, actions: List[int]) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
		
		self.execute_transitions(actions)
		finished = self.is_game_finished()
		rewards = np.array([player.reward for player in self._players])
		timeout = self.is_game_timedout()
		return self.make_obs(), rewards, finished, timeout, {}
	
	def render(self) -> np.ndarray | list[np.ndarray] | None:
		if self._render is None:
			try:
				from .render import Viewer
				self._render = Viewer((self.rows, self.cols), visible=self._show_viewer)
			except Exception as e:
				print('Caught exception %s when trying to import Viewer class.' % str(e.args))
		
		return self._render.render(self, return_rgb_array=(self.render_mode == 'rgb_array'))
	
	def close_render(self):
		if self._render is not None:
			self._render.close()
			self._render = None
	
	def close(self):
		super().close()
		self.close_render()
		