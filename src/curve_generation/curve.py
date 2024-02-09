"""
@file: curve.py
@breif: Trajectory generation
@author: Winter
@update: 2023.5.31
"""
import math
from abc import ABC, abstractmethod

class Curve(ABC):
	def __init__(self, step: float) -> None:
		"""
		Base class for curve generation.

		Parameters:
			step (float): Simulation or interpolation size
		"""
		self.step = step

	@abstractmethod
	def run(self, points: list):
		"""
        Running both generation and animation.
        """
		pass

	@abstractmethod
	def generation(self, start_pose: tuple, goal_pose: tuple):
		"""
		Generate the curve.
		"""
		pass

	def trigonometric(self, alpha: float, beta: float):
		"""
		Calculate some useful trigonometric value with angles.
		"""
		return math.sin(alpha), math.sin(beta), math.cos(alpha), math.cos(beta), \
			math.sin(alpha - beta), math.cos(alpha - beta)

	def pi2pi(self, theta: float) -> float:
		"""
		Truncate the angle to the interval of -π to π.
		"""
		while theta > math.pi:
			theta -= 2.0 * math.pi
		while theta < -math.pi:
			theta += 2.0 * math.pi
		return theta

	def mod2pi(self, theta: float) -> float:
		"""
		Perform modulus operation on 2π.
		"""
		return theta - 2.0 * math.pi * math.floor(theta / math.pi / 2.0)
	
	def length(self, path: list) -> float:
		"""
		Calculate path or trajectory length with `path` format [(ix, iy)] (i from 0 to N)
		"""
		dist = 0
		for i in range(len(path) - 1):
			dist = dist + math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
		return dist