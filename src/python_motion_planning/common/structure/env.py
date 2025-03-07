"""
@file: env.py
@breif: 2-dimension environment
@author: Winter
@update: 2023.1.13
"""
import numpy as np

from abc import ABC
from scipy.spatial import KDTree

class Env(ABC):
	"""
	Class for building 2-d workspace of robots.

	Parameters:
		params (dict): parameters of environment
	"""
	def __init__(self, params: dict) -> None:
		self.x_range, self.y_range = 0, 0

class Grid(Env):
	"""
	Class for discrete 2-d grid map.
	"""
	def __init__(self, params: dict) -> None:
		super().__init__(params)
		if "grid_map" not in params.keys():
			raise RuntimeError("parameters `grid_map` should be configured.")

		self.grid_map, self.obstacles = None, []
		if "grid_map" in params["grid_map"].keys():
			self.grid_map = params["grid_map"]["grid_map"]
			self.x_range = params["grid_map"]["dimensions"][0]
			self.y_range = params["grid_map"]["dimensions"][1]
			obs_indices = np.vstack(np.where(self.grid_map == 1)).T.tolist()
			self.obstacles = [(obs[1], obs[0]) for obs in obs_indices]
		else:
			self.x_range = params["grid_map"]["dimensions"][0]
			self.y_range = params["grid_map"]["dimensions"][1]
			self.grid_map = np.zeros((self.y_range, self.x_range))
			for (ox, oy) in params["grid_map"]["obstacles"]:
				self.obstacles.append((ox, oy))
				self.grid_map[oy, ox] = 1

		# resolution
		self.resolution = params["grid_map"]["resolution"]

		# obstacles
		self.obstacles_tree = KDTree(np.array(self.obstacles))

	def update(self, obstacles):
		self.obstacles = obstacles 
		self.obstacles_tree = KDTree(np.array(list(obstacles)))
		for (ox, oy) in obstacles:
			self.grid_map[oy, ox] = 1
