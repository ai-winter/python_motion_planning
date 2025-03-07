"""
@file: cubic_spline.py
@breif: Cubic spline generation
@author: Winter
@update: 2023.7.28
"""
import math
import bisect
import numpy as np

from typing import List

from .curve import Curve

from python_motion_planning.common.geometry import Point3d

class CubicSpline(Curve):
	"""
	Class for cubic spline generation.

	Parameters:
		step (float): Simulation or interpolation size

	Examples:
		>>> from python_motion_planning.curve_generation import CubicSpline
		>>>	points = [(0, 0, 0), (10, 10, -90), (20, 5, 60)]
		>>> generator = CubicSpline(step)
		>>> generator.run(points)
	"""
	def __init__(self, step: float) -> None:
		super().__init__(step)
	
	def __str__(self) -> str:
		return "Cubic Spline"

	def spline(self, x_list: List[float], y_list: List[float], t: List[float]):
		"""
		Running both generation and animation.

		Parameters:
			x_list (list[float]): path points x-direction
			y_list (list[float]): path points y-direction
			t (list[float]): parameter
		
		Returns:
			p (list[float]): The (x, y) of curve with given t
			dp (list[float]): The derivative (dx, dy) of curve with given t
		"""
		# cubic polynomial functions
		a, b, c, d = y_list, [], [], []
		h = np.diff(x_list)
		num = len(x_list)

		# calculate coefficient matrix
		A = np.zeros((num, num))
		for i in range(1, num - 1):
			A[i, i - 1] = h[i - 1]
			A[i, i] = 2.0 * (h[i - 1] + h[i])
			A[i, i + 1] = h[i]
		A[0, 0] = 1.0
		A[num - 1, num - 1] = 1.0

		B = np.zeros(num)
		for i in range(1, num - 1):
			B[i] = 3.0 * (a[i + 1] - a[i]) / h[i] - \
					3.0 * (a[i] - a[i - 1]) / h[i - 1]

		c = np.linalg.solve(A, B)
		for i in range(num - 1):
			d.append((c[i + 1] - c[i]) / (3.0 * h[i]))
			b.append((a[i + 1] - a[i]) / h[i] - h[i] * (c[i + 1] + 2.0 * c[i]) / 3.0)

		# calculate spline value and its derivative
		p, dp = [], []
		for it in t:
			if it < x_list[0] or it > x_list[-1]:
				continue
			i = bisect.bisect(x_list, it) - 1
			dx = it - x_list[i]
			p.append(a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3)
			dp.append(b[i] + 2.0 * c[i] * dx + 3.0 * d[i] * dx**2)

		return p, dp

	def generation(self, start_pose: Point3d, goal_pose: Point3d) -> None:
		pass

	def run(self, points: List[Point3d]):
		"""
		Running both generation and animation.

		Parameters:
			points (list[Point3d]): path points
		"""
		assert len(points) >= 2, "Number of points should be at least 2."

		x_list = [pt.x() for pt in points]
		y_list = [pt.y() for pt in points]
		dx, dy = np.diff(x_list), np.diff(y_list)
		ds = [math.hypot(idx, idy) for (idx, idy) in zip(dx, dy)]
		s = [0]
		s.extend(np.cumsum(ds))
		t = np.arange(0, s[-1], self.step)

		path_x, d_path_x = self.spline(s, x_list, t)
		path_y, d_path_y = self.spline(s, y_list, t)
		path_yaw = [math.atan2(d_path_y[i], d_path_x[i]) for i in range(len(d_path_x))]

		return [path_x, path_y, path_yaw], [
			{"type": "path", "name": "normal", "props": {"style": "-", "color": "#1f77b4"},
			 "data": [(ix, iy) for (ix, iy) in zip(path_x, path_y)],
			},
			{"type": "marker", "name": "normal", "props": {"marker": "x", "color": "#ff0000", "size": 40},
			 "data": [[pt.x() for pt in points], [pt.y() for pt in points]]
			}
		]