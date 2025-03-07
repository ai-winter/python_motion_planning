"""
@file: curve.py
@breif: Trajectory generation
@author: Winter
@update: 2023.5.31
"""
import math

from typing import List
from abc import ABC, abstractmethod

from python_motion_planning.common.geometry import Point3d

class Curve(ABC):
	def __init__(self, step: float) -> None:
		"""
		Base class for curve generation.

		Parameters:
			step (float): Simulation or interpolation size
		"""
		self.step = step

	@abstractmethod
	def run(self, points: List[Point3d]):
		"""
        Running both generation and animation.
        """
		pass

	@abstractmethod
	def generation(self, start_pose: Point3d, goal_pose: Point3d):
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
	
	def length(self, path: List[Point3d]) -> float:
		"""
		Calculate path or trajectory length with `path` format [(ix, iy)] (i from 0 to N)
		"""
		dist = 0
		for i in range(len(path) - 1):
			dist = dist + math.hypot(path[i + 1].x() - path[i].x(), path[i + 1].y() - path[i].y())
		return dist