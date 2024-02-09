"""
@file: bezier_curve.py
@breif: Bezier curve generation
@author: Winter
@update: 2023.7.25
"""
import numpy as np

from scipy.special import comb
from python_motion_planning.utils import Plot
from .curve import Curve

class Bezier(Curve):
	"""
	Class for Bezier curve generation.

	Parameters:
		step (float): Simulation or interpolation size
		offset (float): The offset of control points

	Examples:
		>>> from python_motion_planning.curve_generation import Bezier
		>>>	points = [(0, 0, 0), (10, 10, -90), (20, 5, 60)]
		>>> generator = Bezier(step, offset)
		>>> generator.run(points)
	"""
	def __init__(self, step: float, offset: float) -> None:
		super().__init__(step)
		self.offset = offset

	def __str__(self) -> str:
		return "Bezier Curve"

	def generation(self, start_pose: tuple, goal_pose: tuple):
		"""
		Generate the Bezier Curve.

		Parameters:
			start_pose (tuple): Initial pose (x, y, yaw)
			goal_pose (tuple): Target pose (x, y, yaw)

		Returns:
			x_list (list): x of the trajectory
			y_list (list): y of the trajectory
			yaw_list (list): yaw of the trajectory
		"""
		sx, sy, _ = start_pose
		gx, gy, _ = goal_pose
		n_points = int(np.hypot(sx - gx, sy - gy) / self.step)
		control_points = self.getControlPoints(start_pose, goal_pose)

		return [self.bezier(t, control_points) for t in np.linspace(0, 1, n_points)], \
			   control_points

	def bezier(self, t: float, control_points: list) ->np.ndarray:
		"""
		Calculate the Bezier curve point.

		Parameters:
			t (float): scale factor
			control_points (list[tuple]): control points

		Returns:
			point (np.array): point in Bezier curve with t
		"""
		n = len(control_points) - 1
		control_points = np.array(control_points)
		return np.sum([comb(n, i) * t ** i * (1 - t) ** (n - i) *
			control_points[i] for i in range(n + 1)], axis=0)

	def getControlPoints(self, start_pose: tuple, goal_pose: tuple):
		"""
		Calculate control points heuristically.

		Parameters:
			start_pose (tuple): Initial pose (x, y, yaw)
			goal_pose (tuple): Target pose (x, y, yaw)

		Returns:
			control_points (list[tuple]): Control points
		"""
		sx, sy, syaw = start_pose
		gx, gy, gyaw = goal_pose

		dist = np.hypot(sx - gx, sy - gy) / self.offset
		return [(sx, sy),
				(sx + dist * np.cos(syaw), sy + dist * np.sin(syaw)),
				(gx - dist * np.cos(gyaw), gy - dist * np.sin(gyaw)),
				(gx, gy)]

	def run(self, points: list):
		"""
        Running both generation and animation.

		Parameters:
			points (list[tuple]): path points
        """
		assert len(points) >= 2, "Number of points should be at least 2."
		import matplotlib.pyplot as plt

		# generation
		path_x, path_y = [], []
		path_control_x, path_control_y = [], []
		for i in range(len(points) - 1):
			path, control_points = self.generation(
				(points[i][0], points[i][1], np.deg2rad(points[i][2])),
				(points[i + 1][0], points[i + 1][1], np.deg2rad(points[i + 1][2])))

			for pt in path:
				path_x.append(pt[0])
				path_y.append(pt[1])

			path_control_x.append(points[i][0])
			path_control_y.append(points[i][1])

			for pt in control_points:
				path_control_x.append(pt[0])
				path_control_y.append(pt[1])

		# animation
		plt.figure("curve generation")
		plt.plot(path_x, path_y, linewidth=2, c="#1f77b4")
		plt.plot(path_control_x, path_control_y, '--o', c='#dddddd', label="Control Points")
		for x, y, theta in points:
			Plot.plotArrow(x, y, np.deg2rad(theta), 2, 'blueviolet')

		plt.axis("equal")
		plt.legend()
		plt.title(str(self))
		plt.show()

