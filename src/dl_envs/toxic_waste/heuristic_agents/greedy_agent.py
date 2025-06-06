#! /usr/bin/env python

import numpy as np

from dl_envs.toxic_waste.toxic_waste_env_v2 import Actions, CellEntity, ActionDirection, ToxicWasteEnvV2, AgentType, WasteType
from dl_envs.toxic_waste.toxic_waste_env_v1 import ToxicWasteEnvV1
from typing import List, Tuple, Dict, Union
from enum import IntEnum
from math import inf, sqrt


SAFE_DISTANCE = 2


class WasteStatus(IntEnum):
	DROPPED = 0
	PICKED = 1
	DISPOSED = 2


class HumanStatus(IntEnum):
	HANDS_FREE = 0
	WASTE_PICKED = 1


class PosNode(object):
	
	_pos: Tuple[int, int]
	_cost: int
	_parent: 'PosNode'
	
	def __init__(self, pos: Tuple[int, int], cost: int, parent_node: 'PosNode'):
		self._pos = pos
		self._cost = cost
		self._parent = parent_node
	
	@property
	def pos(self) -> Tuple[int, int]:
		return self._pos
	
	@property
	def cost(self) -> int:
		return self._cost
	
	@property
	def parent(self) -> 'PosNode':
		return self._parent
	
	@cost.setter
	def cost(self, new_cost: int) -> None:
		self._cost = new_cost
		
	def __str__(self):
		return 'Pos (%d, %d) with cost: %d' % (*self._pos, self._cost)


def get_adj_pos(pos: Tuple[int, int]) -> List[Tuple[int, int]]:
	return [(pos[0] - 1, pos[1]), (pos[0] + 1, pos[1]), (pos[0], pos[1] - 1), (pos[0], pos[1] + 1)]


class GreedyAgent(object):
	
	_pos: Tuple[int, int]
	_orientation: Tuple[int, int]
	_agent_id: str
	_status: HumanStatus
	_nxt_waste_idx: int
	_waste_pos: Dict[int, Tuple[int, int]]
	_waste_order: List[int] = None
	_rng_gen: np.random.Generator
	_map_adjacencies: Dict[Tuple[int, int], List[Tuple[int, int]]]
	_version: int
	_agent_type: int
	_plan: str
	
	def __init__(self, pos_init: Tuple[int, int], orient_init: Tuple[int, int], agent_id: str, objs_pos: Dict[int, Tuple[int, int]],
				 rng_seed: int, field: np.ndarray, version: int, door_pos: Tuple[int, int] = (-1, -1), agent_type: int = AgentType.HUMAN):
		
		self._pos = pos_init
		self._orientation = orient_init
		self._agent_id = agent_id
		self._status = HumanStatus.HANDS_FREE
		self._nxt_waste_idx = -1
		self._waste_pos = objs_pos.copy()
		self._rng_gen = np.random.default_rng(rng_seed)
		self._version = version
		self._door_pos = door_pos
		self._agent_type = agent_type
		self._plan = 'none' if agent_type == AgentType.ROBOT else 'collect'
		
		# Create adjacency map
		rows, cols = field.shape
		free_pos = [(row, col) for row in range(rows) for col in range(cols)
					if field[row, col] == CellEntity.EMPTY or field[row, col] == CellEntity.ICE or field[row, col] == CellEntity.TOXIC]
		free_pos.append(door_pos)
		free_pos.sort()
		self._map_adjacencies = {}
		for pos in free_pos:
			adj_key = (pos[0], pos[1])
			adjacencies = []
			adjs_pos = [(pos[0] - 1, pos[1]), (pos[0] + 1, pos[1]), (pos[0], pos[1] - 1), (pos[0], pos[1] + 1)]
			for a_pos in adjs_pos:
				if a_pos in free_pos:
					adjacencies += [(a_pos[0], a_pos[1])]
			self._map_adjacencies[adj_key] = adjacencies
	
	@property
	def pos(self) -> Tuple[int, int]:
		return self._pos
	
	@property
	def orientation(self) -> Tuple[int, int]:
		return self._orientation
	
	@property
	def agent_id(self) -> str:
		return self._agent_id
	
	@property
	def status(self) -> HumanStatus:
		return self._status
	
	@property
	def next_waste_idx(self) -> int:
		return self._nxt_waste_idx

	@property
	def waste_order(self) -> List[int]:
		return self._waste_order

	@property
	def waste_pos(self) -> Dict[int, Tuple[int, int]]:
		return self._waste_pos

	@property
	def plan(self) -> str:
		return self._plan

	@pos.setter
	def pos(self, new_pos: Tuple[int, int]) -> None:
		self._pos = new_pos
	
	@status.setter
	def status(self, new_status: HumanStatus) -> None:
		self._status = new_status
	
	@next_waste_idx.setter
	def next_waste_idx(self, nxt_obj: int) -> None:
		self._nxt_waste_idx = nxt_obj
		
	@waste_order.setter
	def waste_order(self, new_order: List[int]) -> None:
		self._waste_order = new_order.copy()
	
	def __repr__(self):
		return 'Greedy agent {} at {} facing {} with objective {} and with {}'.format(self._agent_id, self._pos, self._orientation,
																					  self._waste_pos[self._nxt_waste_idx], HumanStatus(self._status).name)
		
	@staticmethod
	def are_facing(h_or: Tuple[int, int], r_or: Tuple[int, int]) -> bool:
		return (h_or[0] + r_or[0]) == 0 and (h_or[1] + r_or[1]) == 0
	
	@staticmethod
	def distance(ptr1: Tuple[int, int], ptr2: Tuple[int, int]) -> float:
		return sqrt((ptr1[0] - ptr2[0]) ** 2 + (ptr1[1] - ptr2[1]) ** 2)
	
	def find_closest_waste(self, wastes_pos: List[Tuple[int, int]]) -> int:
		
		dist = inf
		closest_waste_idxs = []
		for idx in range(len(wastes_pos)):
			pos = wastes_pos[idx]
			dist_tmp = sqrt((self._pos[0] - pos[0])**2 + (self._pos[1] - pos[1])**2)
			if dist_tmp < dist:
				closest_waste_idxs = [idx]
				dist = dist_tmp
			elif dist_tmp == dist:
				closest_waste_idxs.append(idx)
		
		return self._rng_gen.choice(closest_waste_idxs) if len(closest_waste_idxs) > 1 else closest_waste_idxs[0]
	
	def reset(self, waste_order: List[int], n_spawn_objs: int, objs_pos: Dict[int, Tuple[int, int]], pick_all: bool = False) -> None:
		self._nxt_waste_idx = -1
		self._status = HumanStatus.HANDS_FREE
		if pick_all:
			self._waste_order = waste_order.copy()
		else:
			if n_spawn_objs > 0:
				self._waste_order = waste_order.copy()
			else:
				self._waste_order = []
		self._waste_pos = objs_pos.copy()
		self._plan = 'none' if self._agent_type == AgentType.ROBOT else 'collect'
	
	def act_human(self, robot_agents: List, objs: List, n_waste_left: int, only_movement: bool = False, problem_type: str = 'full') -> int:
	
		robot_pos = robot_agents[0].position
		robot_or = robot_agents[0].orientation
		# print('Problem type: ', problem_type)
		# print('Agent HUMAN has plan ' + self._plan)
		# print('Sequence: ', self._waste_order, '\tNext waste: ', self._nxt_waste_idx)

		if (only_movement or n_waste_left <= 0 or
				(problem_type == 'pick_one' and any([obj.hold_state == WasteStatus.DISPOSED for obj in objs]))):
			self._plan = 'exit'
			nxt_waste = self._door_pos
			return int(self.move_to_position(nxt_waste))

		elif problem_type == 'move_catch' and any([obj.was_picked for obj in objs]):
			if self._status == HumanStatus.HANDS_FREE:
				self._plan = 'exit'
				nxt_waste = self._door_pos
				return int(self.move_to_position(nxt_waste))
			else:
				return int(Actions.INTERACT)

		else:
			if self._nxt_waste_idx < 0:
				self._nxt_waste_idx = self._waste_order.pop(0)

			if self._status == HumanStatus.HANDS_FREE:
				nxt_waste = self._waste_pos[self._nxt_waste_idx]
				found_waste = False
				for obj in objs:
					waste_pos = obj.position
					if waste_pos == nxt_waste:
						found_waste = True
						break
					else:
						waste_adj_pos = [(waste_pos[0] - 1, waste_pos[1]), (waste_pos[0] + 1, waste_pos[1]),
										 (waste_pos[0], waste_pos[1] - 1), (waste_pos[0], waste_pos[1] + 1)]
						if nxt_waste in waste_adj_pos:
							self._waste_pos[self._nxt_waste_idx] = waste_pos
							nxt_waste = waste_pos
							found_waste = True
							break
				
				if not found_waste:
					if self._version == 1 or (self._version == 2 and len(self._waste_order) > 0):
						self._nxt_waste_idx = self._waste_order.pop(0)
						nxt_waste = self._waste_pos[self._nxt_waste_idx]
					
					else:
						if self._version == 2:
							self._plan = 'exit'
							nxt_waste = self._door_pos
				
				human_adj_pos = get_adj_pos(self._pos)
				if nxt_waste == self._door_pos:
					return int(self.move_to_position(nxt_waste))
				
				else:
					if nxt_waste in human_adj_pos and nxt_waste[0] == (self._pos[0] + self._orientation[0]) and nxt_waste[1] == (self._pos[1] + self._orientation[1]):
						return int(Actions.INTERACT)
					
					else:
						return int(self.move_to_position(nxt_waste))
			
			else:
				if problem_type == 'move_catch':
					return int(Actions.INTERACT)

				else:
					self._waste_pos[self._nxt_waste_idx] = self._pos

					if robot_pos == (self._pos[0] + self._orientation[0], self._pos[1] + self._orientation[1]):
						if self.are_facing(self._orientation, robot_or):
							return int(Actions.INTERACT)
						else:
							return int(Actions.STAY)

					else:
						return int(self.move_to_position(robot_pos))
	
	def act_robot(self, robot_agents: List, human_agents: List, objs: List) -> int:
		
		def shadow_human() -> int:
			distance = self.distance(self._pos, human_pos)
			if abs(distance) < SAFE_DISTANCE:
				h_direction = ((human_pos[0] - self._pos[0]) * -1, human_pos[1] - self._pos[1] * -1)
				if self._pos in list(self._map_adjacencies.keys()):
					if (self._pos[0] + h_direction[0], self._pos[1] + h_direction[1]) in self._map_adjacencies[self._pos]:
						if h_direction == ActionDirection.UP.value:
							return int(Actions.UP)
						elif h_direction == ActionDirection.DOWN.value:
							return int(Actions.DOWN)
						elif h_direction == ActionDirection.LEFT.value:
							return int(Actions.LEFT)
						else:
							return int(Actions.RIGHT)
					elif (self._pos[0] + self._orientation[0] * -1, self._pos[1] + self._orientation[1] * -1) in self._map_adjacencies[self._pos]:
						if self._orientation == ActionDirection.UP.value:
							return int(Actions.DOWN)
						elif self._orientation == ActionDirection.DOWN.value:
							return int(Actions.UP)
						elif self._orientation == ActionDirection.LEFT.value:
							return int(Actions.RIGHT)
						else:
							return int(Actions.LEFT)
					else:
						return int(self._rng_gen.choice(list(Actions)))
				else:
					return int(self._rng_gen.choice(list(Actions)))
			elif abs(distance) == SAFE_DISTANCE:
				return int(Actions.STAY)
			else:
				return int(self.move_to_position(human_pos))
		
		human_pos = human_agents[0].position
		human_or = human_agents[0].orientation
		human_holding = (human_agents[0].held_objects is not None and len(human_agents[0].held_objects) > 0)
		# print('Agent ASTRO has plan ' + self._plan)
		for idx in range(len(objs)):
			self.waste_pos[idx] = objs[idx].position
		
		if human_holding:
			self._plan = 'disposal'
			if abs(self.distance(self._pos, human_pos)) == 1:
				if self.are_facing(human_or, self._orientation):
					return int(Actions.INTERACT)
				else:
					return int(self.move_to_position(human_pos))
			
			elif abs(self.distance(self._pos, human_pos)) <= SAFE_DISTANCE + 1:
				return int(Actions.STAY)
			
			else:
				return int(self.move_to_position(human_pos))
		
		else:
			if self._plan == 'shadow':
				return shadow_human()
			
			elif self._plan == 'identify':
				waste_pos = self.waste_pos[self._nxt_waste_idx]
				if abs(self.distance(self._pos, waste_pos)) == 1 and (self._pos[0] + self._orientation[0], self._pos[1] + self._orientation[1]) == waste_pos:
					self._plan = 'none'
					return int(Actions.IDENTIFY)
				else:
					return int(self.move_to_position(waste_pos))
				
			else:
				if self._version == 1:
					self._plan = 'shadow'
					return shadow_human()
				
				else:
					waste_unidentified = [(idx, objs[idx].position) for idx in range(len(objs)) if not objs[idx].identified]
					if len(waste_unidentified) < 1 or self._rng_gen.random() < 0.5:
						self._plan = 'shadow'
						return shadow_human()
					else:
						waste_idx = [waste[0] for waste in waste_unidentified]
						wastes_pos = [waste[1] for waste in waste_unidentified]
						self._nxt_waste_idx = waste_idx[self.find_closest_waste(wastes_pos)]
						if (abs(self.distance(self._pos, self.waste_pos[self._nxt_waste_idx])) == 1 and
								(self._pos[0] + self._orientation[0], self._pos[1] + self._orientation[1]) == self.waste_pos[self._nxt_waste_idx]):
							return int(Actions.IDENTIFY)
						else:
							self._plan = 'identify'
							return int(self.move_to_position(self.waste_pos[self._nxt_waste_idx]))
		
	def act(self, obs: Union[ToxicWasteEnvV1.Observation, ToxicWasteEnvV2.Observation], only_movement: bool = False, problem_type: str = 'full_problem') -> int:
		
		self_agent = [agent for agent in obs.players if agent.name == self._agent_id][0]
		robots = [agent for agent in obs.players if agent.agent_type == AgentType.ROBOT]
		humans = [agent for agent in obs.players if agent.agent_type == AgentType.HUMAN]
		objs = obs.objects
		n_waste_left = 0
		for obj in objs:
			if problem_type == 'move_catch':
				if not obj.was_picked:
					n_waste_left += 1
			elif problem_type == 'only_green':
				if obj.waste_type == WasteType.GREEN and obj.hold_state != WasteStatus.DISPOSED:
					n_waste_left += 1
			elif problem_type == 'green_yellow':
				if obj.hold_state != WasteStatus.DISPOSED and (obj.waste_type == WasteType.GREEN or obj.waste_type == WasteType.YELLOW):
					n_waste_left += 1
			else:
				if obj.hold_state != WasteStatus.DISPOSED:
					n_waste_left += 1
		self._pos = self_agent.position
		self._orientation = self_agent.orientation
		self._status = HumanStatus.WASTE_PICKED if self_agent.is_holding_object() else HumanStatus.HANDS_FREE
		if self._agent_type == AgentType.ROBOT:
			return self.act_robot(robots, humans, objs)
		else:
			return self.act_human(robots, objs, n_waste_left, only_movement, problem_type)
	
	def expand_pos(self, start_node: PosNode, objective_pos: Tuple[int, int]) -> Tuple[int, int]:
		
		node_pos = start_node.pos
		cost = start_node.cost
		
		pos_adj = get_adj_pos(node_pos)
		if objective_pos in pos_adj:
			return objective_pos
		
		else:
			seen_nodes = [start_node]
			nodes_visit = []
			for nxt_pos in self._map_adjacencies[node_pos]:
				nodes_visit += [PosNode(nxt_pos, cost + 1, None)]
				seen_nodes += [nodes_visit[-1]]
			
			nxt_node = None
			done = False
			while not done:
				nxt_node = nodes_visit.pop(0)
				nxt_node_adj = get_adj_pos(nxt_node.pos)
				if objective_pos in nxt_node_adj:
					done = True
				else:
					seen_pos = [node.pos for node in seen_nodes]
					for pos in self._map_adjacencies[nxt_node.pos]:
						if pos not in seen_pos:
							nodes_visit += [PosNode(pos, nxt_node.cost + 1, nxt_node)]
							seen_nodes += [nodes_visit[-1]]
							
			while nxt_node.parent is not None:
				nxt_node = nxt_node.parent
			
			return nxt_node.pos
	
	def move_to_position(self, objective_pos: Tuple[int, int]) -> int:
		d_row = objective_pos[0] - self._pos[0]
		d_col = objective_pos[1] - self._pos[1]

		if d_row < 0:
			if d_col < 0:
				action = self._rng_gen.choice([Actions.UP, Actions.LEFT])
			elif d_col > 0:
				action = self._rng_gen.choice([Actions.UP, Actions.RIGHT])
			else:
				action = Actions.UP
		elif d_row > 0:
			if d_col < 0:
				action = self._rng_gen.choice([Actions.DOWN, Actions.LEFT])
			elif d_col > 0:
				action = self._rng_gen.choice([Actions.DOWN, Actions.RIGHT])
			else:
				action = Actions.DOWN
		else:
			if d_col < 0:
				action = Actions.LEFT
			else:
				action = Actions.RIGHT

		nxt_pos = (self._pos[0] + ActionDirection[Actions(action).name].value[0], self._pos[1] + ActionDirection[Actions(action).name].value[1])
		if nxt_pos in self._map_adjacencies[self._pos] and nxt_pos == objective_pos:
			return action
		else:
			best_pos = self.expand_pos(PosNode(self._pos, 1, None), objective_pos)
			mv_direction = (best_pos[0] - self._pos[0], best_pos[1] - self._pos[1])
			if mv_direction[0] == -1 and mv_direction[1] == 0:
				return Actions.UP
			elif mv_direction[0] == 1 and mv_direction[1] == 0:
				return Actions.DOWN
			elif mv_direction[0] == 0 and mv_direction[1] == -1:
				return Actions.LEFT
			else:
				return Actions.RIGHT
