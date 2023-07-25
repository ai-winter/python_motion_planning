'''
@file: curve.py
@breif: Trajectory generation
@author: Winter
@update: 2023.5.31
'''
import math
from abc import ABC, abstractmethod

class Curve(ABC):
	def __init__(self, step: float, max_curv: float) -> None:
		'''
		Base class for curve generation.

		Parameters
		----------
		step: float
			Simulation or interpolation size
		max_curv: float
			The maximum curvature of the curve
		'''
		self.step = step
		self.max_curv = max_curv

	@abstractmethod
	def run(self, points: list):
		'''
        Running both generation and animation.
        '''
		pass

	@abstractmethod
	def generation(self, start: tuple, goal: tuple):
		'''
		Generate the curve.
		'''
		pass

	def trigonometric(self, alpha, beta):
		'''
		Calculate some useful trigonometric value with angles.
		'''        
		return math.sin(alpha), math.sin(beta), math.cos(alpha), math.cos(beta), \
			math.sin(alpha - beta), math.cos(alpha - beta)

	def pi2pi(self, theta):
		'''
		Truncate the angle to the interval of -π to π.
		'''        
		while theta > math.pi:
			theta -= 2.0 * math.pi
		while theta < -math.pi:
			theta += 2.0 * math.pi
		return theta

	def mod2pi(self, theta):
		'''
		Perform modulus operation on 2π.
		'''        
		return theta - 2.0 * math.pi * math.floor(theta / math.pi / 2.0)