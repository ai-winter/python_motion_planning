"""
@file: bezier_curve.py
@breif: Bezier curve generation
@author: Winter
@update: 2023.7.25
"""
import math
import numpy as np

from typing import List
from scipy.special import comb

from .curve import Curve

from python_motion_planning.common.geometry import Point3d

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

	def generation(self, start_pose: Point3d, goal_pose: Point3d):
		"""
		Generate the Bezier Curve.

		Parameters:
			start_pose (Point3d): Initial pose (x, y, yaw)
			goal_pose (Point3d): Target pose (x, y, yaw)

		Returns:
			x_list (list): x of the trajectory
			y_list (list): y of the trajectory
			yaw_list (list): yaw of the trajectory
		"""
		sx, sy, _ = start_pose
		gx, gy, _ = goal_pose
		n_points = int(math.hypot(sx - gx, sy - gy) / self.step)
		control_points = self.getControlPoints(start_pose, goal_pose)

		return [
			self.bezier(t, control_points) for t in np.linspace(0, 1, n_points)
		], control_points

	def bezier(self, t: float, control_points: List[Point3d]) -> Point3d:
		"""
		Calculate the Bezier curve point.

		Parameters:
			t (float): scale factor
			control_points (list[Point3d]): control points

		Returns:
			point (np.array): point in Bezier curve with t
		"""
		n = len(control_points) - 1

		pt_x, pt_y = 0.0, 0.0
		for i in range(n + 1):
			factor = comb(n, i) * t ** i * (1 - t) ** (n - i)
			pt_x += factor * control_points[i].x()
			pt_y += factor * control_points[i].y()
		return Point3d(pt_x, pt_y)

	def getControlPoints(self, start_pose: Point3d, goal_pose: Point3d) -> List[Point3d]:
		"""
		Calculate control points heuristically.

		Parameters:
			start_pose (Point3d): Initial pose (x, y, yaw)
			goal_pose (Point3d): Target pose (x, y, yaw)

		Returns:
			control_points (list[tuple]): Control points
		"""
		sx, sy, syaw = start_pose
		gx, gy, gyaw = goal_pose

		dist = math.hypot(sx - gx, sy - gy) / self.offset
		return [
			Point3d(sx, sy),
			Point3d(sx + dist * math.cos(syaw), sy + dist * math.sin(syaw)),
			Point3d(gx - dist * math.cos(gyaw), gy - dist * math.sin(gyaw)),
			Point3d(gx, gy)
		]

	def run(self, points: List[Point3d]):
		"""
        Running both generation and animation.

		Parameters:
			points (list[Point3d]): path points
        """
		assert len(points) >= 2, "Number of points should be at least 2."

		# generation
		path_x, path_y = [], []
		path_control_x, path_control_y = [], []
		for i in range(len(points) - 1):
			path, control_points = self.generation(points[i], points[i + 1])
			for pt in path:
				path_x.append(pt[0])
				path_y.append(pt[1])

			path_control_x.append(points[i][0])
			path_control_y.append(points[i][1])

			for pt in control_points:
				path_control_x.append(pt[0])
				path_control_y.append(pt[1])

		return [path_x, path_y], [
			{"type": "path", "name": "normal", "props": {"style": "-", "color": "#1f77b4"},
			 "data": [(ix, iy) for (ix, iy) in zip(path_x, path_y)],
			},
			{"type": "path", "name": "normal", "props": {"style": "--", "color": "#dddddd", "marker": "o"},
			 "data": [(ix, iy) for (ix, iy) in zip(path_control_x, path_control_y)],
			},
			{"type": "marker", "name": "arrow", "props": {"length": 2, "color": "blueviolet"},
			 "data": [(pt.x(), pt.y(), pt.theta()) for pt in points]
			}
		]

class Bernstein:
    def __init__(self) -> None:
        pass

    def __call__(self, n: int) -> np.array:
        if n == 5:
            return np.array([
                [1,   0,   0,   0,  0,  0],
                [-5,   5,   0,   0,  0,  0],
                [10, -20,  10,   0,  0,  0],
                [-10,  30, -30,  10,  0,  0],
                [5, -20,  30, -20,  5,  0],
                [-1,   5, -10,  10, -5,  1]
            ])
        else:
            raise NotImplementedError("order n must be 1~10")