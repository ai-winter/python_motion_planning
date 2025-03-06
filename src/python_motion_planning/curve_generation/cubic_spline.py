"""
@file: cubic_spline.py
@breif: Cubic spline generation
@author: Winter
@update: 2023.7.28
"""
import math
import bisect
import numpy as np

from .curve import Curve

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

	def spline(self, x_list: list, y_list: list, t: list):
		"""
		Running both generation and animation.

		Parameters:
			x_list (list[tuple]): path points x-direction
			y_list (list[tuple]): path points y-direction
			t (list): parameter
		
		Returns:
			p (list): The (x, y) of curve with given t
			dp (list): The derivative (dx, dy) of curve with given t
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

	def generation(self, start_pose: tuple, goal_pose: tuple):
		pass

	def run(self, points: list):
		"""
		Running both generation and animation.

		Parameters:
			points (list[tuple]): path points
		"""
		assert len(points) >= 2, "Number of points should be at least 2."
		import matplotlib.pyplot as plt

		if len(points[0]) == 2:
			x_list = [ix for (ix, _) in points]
			y_list = [iy for (_, iy) in points]
		elif len(points[0]) == 3:
			x_list = [ix for (ix, _, _) in points]
			y_list = [iy for (_, iy, _) in points]
		else:
			raise NotImplementedError
		
		dx, dy = np.diff(x_list), np.diff(y_list)
		ds = [math.hypot(idx, idy) for (idx, idy) in zip(dx, dy)]
		s = [0]
		s.extend(np.cumsum(ds))
		t = np.arange(0, s[-1], self.step)

		path_x, d_path_x = self.spline(s, x_list, t)
		path_y, d_path_y = self.spline(s, y_list, t)
		path_yaw = [math.atan2(d_path_y[i], d_path_x[i]) for i in range(len(d_path_x))]

		# animation
		plt.figure("curve generation")

		# static
		plt.figure("curve generation")
		plt.plot(path_x, path_y, linewidth=2, c="#1f77b4")
		for x, y, _ in points:
			plt.plot(x, y, "xr", linewidth=2)
		plt.axis("equal")
		plt.title(str(self))

		plt.figure("yaw")
		plt.plot(t, [math.degrees(iyaw) for iyaw in path_yaw], "-r")
		plt.title("yaw curve")

		plt.show()