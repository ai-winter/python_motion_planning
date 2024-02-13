"""
@file: dubins_curve.py
@breif: Dubins curve generation
@author: Winter
@update: 2023.5.31
"""
import math
import numpy as np

from scipy.spatial.transform import Rotation as Rot
from python_motion_planning.utils import Plot
from .curve import Curve

class Dubins(Curve):
	"""
	Class for Dubins curve generation.

	Parameters:
		step (float): Simulation or interpolation size
		max_curv (float): The maximum curvature of the curve

	Examples:
		>>> from python_motion_planning.curve_generation import Dubins
		>>>	points = [(0, 0, 0), (10, 10, -90), (20, 5, 60)]
		>>> generator = Dubins(step, max_curv)
		>>> generator.run(points)

	References:
		[1] On curves of minimal length with a constraint on average curvature, and with prescribed initial and terminal positions and tangents
	"""
	def __init__(self, step: float, max_curv: float) -> None:
		super().__init__(step)
		self.max_curv = max_curv
	
	def __str__(self) -> str:
		return "Dubins Curve"

	def LSL(self, alpha: float, beta: float, dist: float):
		"""
		Left-Straight-Left generation mode.

		Parameters:
			alpha (float): Initial pose of (0, 0, alpha)
			beta (float): Goal pose of (dist, 0, beta)
			dist (float): The distance between the initial and goal pose

		Returns:
			t (float): Moving lenght of segments
			p (float): Moving lenght of segments
			q (float): Moving lenght of segments
			mode (list): Motion mode
		"""
		sin_a, sin_b, cos_a, cos_b, _, cos_a_b  = self.trigonometric(alpha, beta)

		p_lsl = 2 + dist ** 2 - 2 * cos_a_b + 2 * dist * (sin_a - sin_b)
		if p_lsl < 0:
			return None, None, None, ["L", "S", "L"]
		else:
			p_lsl = math.sqrt(p_lsl)

		t_lsl = self.mod2pi(-alpha + math.atan2(cos_b - cos_a, dist + sin_a - sin_b))
		q_lsl = self.mod2pi(beta - math.atan2(cos_b - cos_a, dist + sin_a - sin_b))

		return t_lsl, p_lsl, q_lsl, ["L", "S", "L"]

	def RSR(self, alpha: float, beta: float, dist: float):
		"""
		Right-Straight-Right generation mode.

		Parameters:
			alpha (float): Initial pose of (0, 0, alpha)
			beta (float): Goal pose of (dist, 0, beta)

		Returns:
			t (float): Moving lenght of segments
			p (float): Moving lenght of segments
			q (float): Moving lenght of segments
			mode (list): Motion mode
		"""
		sin_a, sin_b, cos_a, cos_b, _, cos_a_b  = self.trigonometric(alpha, beta)

		p_rsr = 2 + dist ** 2 - 2 * cos_a_b + 2 * dist * (sin_b - sin_a)
		if p_rsr < 0:
			return None, None, None, ["R", "S", "R"]
		else:
			p_rsr = math.sqrt(p_rsr)

		t_rsr = self.mod2pi(alpha - math.atan2(cos_a - cos_b, dist - sin_a + sin_b))
		q_rsr = self.mod2pi(-beta + math.atan2(cos_a - cos_b, dist - sin_a + sin_b))

		return t_rsr, p_rsr, q_rsr, ["R", "S", "R"]

	def LSR(self, alpha: float, beta: float, dist: float):
		"""
		Left-Straight-Right generation mode.

		Parameters:
			alpha (float): Initial pose of (0, 0, alpha)
			beta (float): Goal pose of (dist, 0, beta)
			dist (float): The distance between the initial and goal pose

		Returns:
			t (float): Moving lenght of segments
			p (float): Moving lenght of segments
			q (float): Moving lenght of segments
			mode (list): Motion mode
		"""
		sin_a, sin_b, cos_a, cos_b, _, cos_a_b  = self.trigonometric(alpha, beta)

		p_lsr = -2 + dist ** 2 + 2 * cos_a_b + 2 * dist * (sin_a + sin_b)
		if p_lsr < 0:
			return None, None, None, ["L", "S", "R"]
		else:
			p_lsr = math.sqrt(p_lsr)

		t_lsr = self.mod2pi(-alpha + math.atan2(-cos_a - cos_b, dist + sin_a + sin_b) - math.atan2(-2.0, p_lsr))
		q_lsr = self.mod2pi(-beta + math.atan2(-cos_a - cos_b, dist + sin_a + sin_b) - math.atan2(-2.0, p_lsr))

		return t_lsr, p_lsr, q_lsr, ["L", "S", "R"]


	def RSL(self, alpha: float, beta: float, dist: float):
		"""
		Right-Straight-Left generation mode.

		Parameters:
			alpha (float): Initial pose of (0, 0, alpha)
			beta (float): Goal pose of (dist, 0, beta)
			dist (float): The distance between the initial and goal pose

		Returns:
			t (float): Moving lenght of segments
			p (float): Moving lenght of segments
			q (float): Moving lenght of segments
			mode (list): Motion mode
		"""
		sin_a, sin_b, cos_a, cos_b, _, cos_a_b  = self.trigonometric(alpha, beta)

		p_rsl = -2 + dist ** 2 + 2 * cos_a_b - 2 * dist * (sin_a + sin_b)
		if p_rsl < 0:
			return None, None, None, ["R", "S", "L"]
		else:
			p_rsl = math.sqrt(p_rsl)

		t_rsl = self.mod2pi(alpha - math.atan2(cos_a + cos_b, dist - sin_a - sin_b) + math.atan2(2.0, p_rsl))
		q_rsl = self.mod2pi(beta - math.atan2(cos_a + cos_b, dist - sin_a - sin_b) + math.atan2(2.0, p_rsl))

		return t_rsl, p_rsl, q_rsl, ["R", "S", "L"]


	def RLR(self, alpha: float, beta: float, dist: float):
		"""
		Right-Left-Right generation mode.

		Parameters:
			alpha (float): Initial pose of (0, 0, alpha)
			beta (float): Goal pose of (dist, 0, beta)
			dist (float): The distance between the initial and goal pose

		Returns:
			t (float): Moving lenght of segments
			p (float): Moving lenght of segments
			q (float): Moving lenght of segments
			mode (list): Motion mode
		"""
		sin_a, sin_b, cos_a, cos_b, _, cos_a_b  = self.trigonometric(alpha, beta)

		p_rlr = (6.0 - dist ** 2 + 2.0 * cos_a_b + 2.0 * dist * (sin_a - sin_b)) / 8.0
		if abs(p_rlr) > 1.0:
			return None, None, None, ["R", "L", "R"]
		else:
			p_rlr = self.mod2pi(2 * math.pi - math.acos(p_rlr))
		
		t_rlr = self.mod2pi(alpha - math.atan2(cos_a - cos_b, dist - sin_a + sin_b) + p_rlr / 2.0)
		q_rlr = self.mod2pi(alpha - beta - t_rlr + p_rlr)

		return t_rlr, p_rlr, q_rlr, ["R", "L", "R"]

	def LRL(self, alpha: float, beta: float, dist: float):
		"""
		Left-Right-Left generation mode.

		Parameters:
			alpha (float): Initial pose of (0, 0, alpha)
			beta (float): Goal pose of (dist, 0, beta)
			dist (float): The distance between the initial and goal pose

		Returns:
			t (float): Moving lenght of segments
			p (float): Moving lenght of segments
			q (float): Moving lenght of segments
			mode (list): Motion mode
		"""
		sin_a, sin_b, cos_a, cos_b, _, cos_a_b  = self.trigonometric(alpha, beta)

		p_lrl = (6.0 - dist ** 2 + 2.0 * cos_a_b + 2.0 * dist * (sin_a - sin_b)) / 8.0
		if abs(p_lrl) > 1.0:
			return None, None, None, ["L", "R", "L"]
		else:
			p_lrl = self.mod2pi(2 * math.pi - math.acos(p_lrl))

		t_lrl = self.mod2pi(-alpha + math.atan2(-cos_a + cos_b, dist + sin_a - sin_b) + p_lrl / 2.0)
		q_lrl = self.mod2pi(beta - alpha - t_lrl + p_lrl)

		return t_lrl, p_lrl, q_lrl, ["L", "R", "L"]    

	def interpolate(self, mode: str, length: float, init_pose: tuple):
		"""
		Planning path interpolation.

		Parameters:
			mode (str): motion, e.g., L, S, R
			length (float): Single step motion path length
			init_pose (tuple): Initial pose (x, y, yaw)

		Returns:
			new_pose (tuple): New pose (new_x, new_y, new_yaw) after moving
		"""
		x, y, yaw = init_pose

		if mode == "S":
			new_x   = x + length / self.max_curv * math.cos(yaw)
			new_y   = y + length / self.max_curv * math.sin(yaw)
			new_yaw = yaw
		elif mode == "L":
			new_x   = x + (math.sin(yaw + length) - math.sin(yaw)) / self.max_curv
			new_y   = y - (math.cos(yaw + length) - math.cos(yaw)) / self.max_curv
			new_yaw = yaw + length
		elif mode == "R":
			new_x   = x - (math.sin(yaw - length) - math.sin(yaw)) / self.max_curv
			new_y   = y + (math.cos(yaw - length) - math.cos(yaw)) / self.max_curv
			new_yaw = yaw - length
		else:
			raise NotImplementedError

		return new_x, new_y, new_yaw

	def generation(self, start_pose: tuple, goal_pose: tuple):
		"""
		Generate the Dubins Curve.

		Parameters:
			start_pose (tuple): Initial pose (x, y, yaw)
			goal_pose (tuple): Target pose (x, y, yaw)

		Returns:
			best_cost (float): Best planning path length
			best_mode (list): Best motion modes
			x_list (list): Trajectory of x
			y_list (list): Trajectory of y
			yaw_list (list): Trajectory of yaw
		"""
		sx, sy, syaw = start_pose
		gx, gy, gyaw = goal_pose

		# coordinate transformation
		gx, gy = gx - sx, gy - sy
		theta = self.mod2pi(math.atan2(gy, gx))
		dist = math.hypot(gx, gy) * self.max_curv
		alpha = self.mod2pi(syaw - theta)
		beta = self.mod2pi(gyaw - theta)

		# select the best motion
		planners = [self.LSL, self.RSR, self.LSR, self.RSL, self.RLR, self.LRL]
		best_t, best_p, best_q, best_mode, best_cost = None, None, None, None, float("inf")

		for planner in planners:
			t, p, q, mode = planner(alpha, beta, dist)
			if t is None:
				continue
			cost = (abs(t) + abs(p) + abs(q))
			if best_cost > cost:
				best_t, best_p, best_q, best_mode, best_cost = t, p, q, mode, cost
		
		# interpolation
		segments = [best_t, best_p, best_q]
		points_num = int(sum(segments) / self.step) + len(segments) + 3
		x_list = [0.0 for _ in range(points_num)]
		y_list = [0.0 for _ in range(points_num)]
		yaw_list = [alpha for _ in range(points_num)]

		i = 0
		for mode_, seg_length in zip(best_mode, segments):
			# path increment
			d_length = self.step if seg_length > 0.0 else -self.step
			x, y, yaw = x_list[i], y_list[i], yaw_list[i]
			# current path length
			length = d_length
			while abs(length) <= abs(seg_length):
				i += 1
				x_list[i], y_list[i], yaw_list[i] = self.interpolate(mode_, length, (x, y, yaw))
				length += d_length
			i += 1
			x_list[i], y_list[i], yaw_list[i] = self.interpolate(mode_, seg_length, (x, y, yaw))

		# failed
		if len(x_list) <= 1:
			return None, None, [], [], []

		# remove unused data
		while len(x_list) >= 1 and x_list[-1] == 0.0:
			x_list.pop()
			y_list.pop()
			yaw_list.pop()

		# coordinate transformation
		rot = Rot.from_euler('z', theta).as_matrix()[0:2, 0:2]
		converted_xy = rot @ np.stack([x_list, y_list])
		x_list = converted_xy[0, :] + sx
		y_list = converted_xy[1, :] + sy
		yaw_list = [self.pi2pi(i_yaw + theta) for i_yaw in yaw_list]

		return best_cost, best_mode, x_list, y_list, yaw_list

	def run(self, points: list):
		"""
		Running both generation and animation.

		Parameters:
			points (list[tuple]): path points
		"""
		assert len(points) >= 2, "Number of points should be at least 2."
		import matplotlib.pyplot as plt

		# generation
		path_x, path_y, path_yaw = [], [], []
		for i in range(len(points) - 1):
			_, _, x_list, y_list, yaw_list = self.generation(
				(points[i][0], points[i][1], np.deg2rad(points[i][2])),
				(points[i + 1][0], points[i + 1][1], np.deg2rad(points[i + 1][2])))

			for j in range(len(x_list)):
				path_x.append(x_list[j])
				path_y.append(y_list[j])
				path_yaw.append(yaw_list[j])
		
		# animation
		plt.figure("curve generation")
		plt.plot(path_x, path_y, linewidth=2, c="#1f77b4")
		for x, y, theta in points:
			Plot.plotArrow(x, y, np.deg2rad(theta), 2, 'blueviolet')

		plt.axis("equal")
		plt.title(str(self))
		plt.show()

		

