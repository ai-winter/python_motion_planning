"""
@file: fem_pos_smooth.py
@breif: Fem-Pos Smoother
@author: Winter
@update: 2024.3.23
"""
import osqp
import numpy as np
from scipy import sparse

from .curve import Curve

class FemPosSmoother(Curve):
	"""
	Class for Fem-pos smoother.

	Parameters:

	Examples:
		>>> from python_motion_planning.curve_generation import FemPosSmoother
		>>>	points = [(0, 0, 0), (10, 10, -90), (20, 5, 60)]
		>>> generator = FemPosSmoother(w_smooth, w_length, w_ref, dx_l, dx_u, dy_l, dy_u)
		>>> generator.run(points)
	"""
	def __init__(self, w_smooth: float, w_length: float, w_ref: float,
		dx_l: float, dx_u: float, dy_l: float, dy_u: float) -> None:
		super().__init__(0.1)
		self.w_smooth = w_smooth
		self.w_length = w_length
		self.w_ref = w_ref
		self.dx_l = dx_l
		self.dx_u = dx_u
		self.dy_l = dy_l
		self.dy_u = dy_u

	def __str__(self) -> str:
		return "Fem-pos Smoother"

	def generation(self, start_pose: tuple, goal_pose: tuple):
		pass
	
	def run(self, points: list, display: bool = True):
		"""
		Running both generation and animation.

		Parameters:
			points (list[tuple]): path points
		"""
		assert len(points) >= 3, "Number of points should be at least 3."
		if len(points[0]) == 3:
			points = [(ix, iy) for ix, iy, _ in points]
		import matplotlib.pyplot as plt

		n = len(points)
		P = np.zeros((2 * n, 2 * n))

		X = np.eye(2) * self.w_smooth
		Y = np.eye(2) * self.w_length
		Z = np.eye(2) * self.w_ref

		P[0:2, 0:2] = X + Y + Z
		P[0:2, 2:4] = -2 * X - Y
		P[2:4, 2:4] = 5 * X + 2 * Y + Z
		P[2 * n - 2:2 * n, 2 * n - 2:2 * n] =  X + Y + Z
		P[2 * n - 4:2 * n - 2, 2 * n - 4:2 * n - 2] =  5 * X + 2 * Y + Z
		P[2 * n - 4:2 * n - 2, 2 * n - 2:2 * n] =  -2 * X - Y

		for i in range(2, n - 2):
			P[2 * i:2 * i + 2, 2 * i:2 * i + 2] = 6 * X + 2 * Y + Z
		for i in range(2, n - 1):
			P[2 * i - 2:2 * i, 2 * i:2 * i + 2] = -4 * X - Y
		for i in range(2, n):
			P[2 * i - 4:2 * i - 2, 2 * i:2 * i + 2] = X
		
		A_I = np.eye(2 * n)
		g = np.zeros((2 * n, 1))
		lower = np.zeros((2 * n, 1))
		upper = np.zeros((2 * n, 1))
		for i, p in enumerate(points):
			g[2 * i], g[2 * i + 1] = -2 * self.w_ref * p[0], -2 * self.w_ref * p[1]
			lower[2 * i], lower[2 * i + 1] = p[0] - self.dx_l, p[1] - self.dy_l
			upper[2 * i], upper[2 * i + 1] = p[0] + self.dx_u, p[1] + self.dy_u

		# solve
		solver = osqp.OSQP()
		solver.setup(sparse.csc_matrix(P), g, sparse.csc_matrix(A_I), lower, upper, verbose=False)
		res = solver.solve()
		opt = res.x

		path_x, path_y = [], []
		for i in range(n):
			path_x.append(opt[2 * i])
			path_y.append(opt[2 * i + 1])

		if display:
			plt.figure("curve generation")
			plt.plot(path_x, path_y, linewidth=2, c="#ff0000", marker="o", label="smooth path")
			raw_x, raw_y = [], []
			for i, (x, y) in enumerate(points):
				# label = "bounding box" if i == 0 else None
				# plt.gca().add_patch(
				# 	plt.Rectangle(xy=(x - self.dx_l, y - self.dy_l), width=self.dx_u + self.dx_l,
				# 	height=self.dy_u + self.dy_l, color='red', linestyle="--", fill=False, label=label)
				# )
				raw_x.append(x)
				raw_y.append(y)
			plt.plot(raw_x, raw_y, linewidth=2, c="#1f77b4", marker="x", label="raw path")
			# plt.axis("equal")
			plt.legend()
			plt.title(str(self))

			plt.show()
		
		return [(ix, iy) for (ix, iy) in zip(path_x, path_y)]