"""
@file: reeds_shepp.py
@breif: Reeds shepp curve generation
@author: Winter
@update: 2023.7.26
"""
import math
import numpy as np

from python_motion_planning.utils import Plot
from .curve import Curve

class ReedsShepp(Curve):
	"""
	Class for Reeds shepp curve generation.

	Parameters:
		step (float): Simulation or interpolation size
		max_curv (float): The maximum curvature of the curve

	Examples:
		>>> from python_motion_planning.curve_generation import ReedsShepp
		>>>	points = [(0, 0, 0), (10, 10, -90), (20, 5, 60)]
		>>> generator = ReedsShepp(step, max_curv)
		>>> generator.run(points)

	References:
		[1] Optimal paths for a car that goes both forwards and backwards
	"""
	def __init__(self, step: float, max_curv: float) -> None:
		super().__init__(step)
		self.max_curv = max_curv
	
	def __str__(self) -> str:
		return "Reeds Shepp Curve"

	def R(self, x, y):
		"""
		Return the polar coordinates (r, theta) of the point (x, y)
		i.e. rcos(theta) = x; rsin(theta) = y

		Parameters:
			x (float): x-coordinate value
			y (float): y-coordinate value

		Returns:
			r, theta (float): Polar coordinates

		"""
		r = math.hypot(x, y)
		theta = math.atan2(y, x)

		return r, theta

	def M(self, theta):
		"""
		Truncate the angle to the interval of -π to π.

		Parameters:
			theta (float): Angle value

		Returns:
			theta (float): Truncated angle value
		"""
		return self.pi2pi(theta)

	class Path:
		"""
		class for Path element
		"""
		def __init__(self, lengths: list = [], ctypes: list = [], x: list = [],
			y: list = [], yaw: list = [], dirs: list = []):
			self.lengths = lengths  	# lengths of each part of path (+: forward, -: backward)
			self.ctypes = ctypes  		# type of each part of the path
			self.path_length = sum([abs(i) for i in lengths])  # total path length
			self.x = x  				# x-coordinate value of curve
			self.y = y  				# y-coordinate value of curve
			self.yaw = yaw  			# yaw value of curve
			self.dirs = dirs 			# direction value of curve (1: forward, -1: backward)

	def SLS(self, x: float, y: float, phi: float):
		"""
		Straight-Left-Straight generation mode.
		"""
		phi = self.M(phi)

		if y > 0.0 and 0.0 < phi < math.pi * 0.99:
			xd = -y / math.tan(phi) + x
			t = xd - math.tan(phi / 2.0)
			u = phi
			v = math.sqrt((x - xd) ** 2 + y ** 2) - math.tan(phi / 2.0)
			return True, t, u, v
		elif y < 0.0 and 0.0 < phi < math.pi * 0.99:
			xd = -y / math.tan(phi) + x
			t = xd - math.tan(phi / 2.0)
			u = phi
			v = -math.sqrt((x - xd) ** 2 + y ** 2) - math.tan(phi / 2.0)
			return True, t, u, v

		return False, 0.0, 0.0, 0.0

	def LRL(self, x: float, y: float, phi: float):
		"""
		Left-Right-Left generation mode. (L+R-L-)
		"""
		r, theta = self.R(x - math.sin(phi), y - 1.0 + math.cos(phi))

		if r <= 4.0:
			u = -2.0 * math.asin(0.25 * r)
			t = self.M(theta + 0.5 * u + math.pi)
			v = self.M(phi - t + u)

			if t >= 0.0 and u <= 0.0:
				return True, t, u, v

		return False, 0.0, 0.0, 0.0

	def LSL(self, x: float, y: float, phi: float):
		"""
		Left-Straight-Left generation mode. (L+S+L+)
		"""
		u, t = self.R(x - math.sin(phi), y - 1.0 + math.cos(phi))

		if t >= 0.0:
			v = self.M(phi - t)
			if v >= 0.0:
				return True, t, u, v

		return False, 0.0, 0.0, 0.0

	def LSR(self, x: float, y: float, phi: float):
		"""
		Left-Straight-Right generation mode. (L+S+R+)
		"""
		r, theta = self.R(x + math.sin(phi), y - 1.0 - math.cos(phi))
		r = r ** 2

		if r >= 4.0:
			u = math.sqrt(r - 4.0)
			t = self.M(theta + math.atan2(2.0, u))
			v = self.M(t - phi)

			if t >= 0.0 and v >= 0.0:
				return True, t, u, v

		return False, 0.0, 0.0, 0.0

	def LRLRn(self, x: float, y: float, phi: float):
		"""
		Left-Right(beta)-Left(beta)-Right generation mode. (L+R+L-R-)
		"""
		xi = x + math.sin(phi)
		eta = y - 1.0 - math.cos(phi)
		rho = 0.25 * (2.0 + math.sqrt(xi * xi + eta * eta))

		if rho <= 1.0:
			u = math.acos(rho)
			t, v = self._calTauOmega(u, -u, xi, eta, phi)
			if t >= 0.0 and v <= 0.0:
				return True, t, u, v

		return False, 0.0, 0.0, 0.0

	def LRLRp(self, x: float, y: float, phi: float):
		"""
		Left-Right(beta)-Left(beta)-Right generation mode. (L+R-L-R+)
		"""
		xi = x + math.sin(phi)
		eta = y - 1.0 - math.cos(phi)
		rho = (20.0 - xi * xi - eta * eta) / 16.0

		if 0.0 <= rho <= 1.0:
			u = -math.acos(rho)
			if u >= -0.5 * math.pi:
				t, v = self._calTauOmega(u, u, xi, eta, phi)
				if t >= 0.0 and v >= 0.0:
					return True, t, u, v

		return False, 0.0, 0.0, 0.0

	def LRSR(self, x: float, y: float, phi: float):
		"""
		Left-Right(pi/2)-Straight-Right generation mode. (L+R-S-R-)
		"""
		xi = x + math.sin(phi)
		eta = y - 1.0 - math.cos(phi)
		rho, theta = self.R(-eta, xi)

		if rho >= 2.0:
			t = theta
			u = 2.0 - rho
			v = self.M(t + 0.5 * math.pi - phi)
			if t >= 0.0 and u <= 0.0 and v <= 0.0:
				return True, t, u, v

		return False, 0.0, 0.0, 0.0

	def LRSL(self, x: float, y: float, phi: float):
		"""
		Left-Right(pi/2)-Straight-Left generation mode. (L+R-S-L-)
		"""
		xi = x - math.sin(phi)
		eta = y - 1.0 + math.cos(phi)
		rho, theta = self.R(xi, eta)

		if rho >= 2.0:
			r = math.sqrt(rho * rho - 4.0)
			u = 2.0 - r
			t = self.M(theta + math.atan2(r, -2.0))
			v = self.M(phi - 0.5 * math.pi - t)
			if t >= 0.0 and u <= 0.0 and v <= 0.0:
				return True, t, u, v

		return False, 0.0, 0.0, 0.0

	def LRSLR(self, x: float, y: float, phi: float):
		"""
		Left-Right(pi/2)-Straight-Left(pi/2)-Right generation mode. (L+R-S-L-R+)
		"""
		xi = x + math.sin(phi)
		eta = y - 1.0 - math.cos(phi)
		r, _ = self.R(xi, eta)

		if r >= 2.0:
			u = 4.0 - math.sqrt(r * r - 4.0)
			if u <= 0.0:
				t = self.M(math.atan2((4.0 - u) * xi - 2.0 * eta, -2.0 * xi + (u - 4.0) * eta))
				v = self.M(t - phi)

				if t >= 0.0 and v >= 0.0:
					return True, t, u, v

		return False, 0.0, 0.0, 0.0

	def SCS(self, x: float, y: float, phi: float):
		"""
		# 2
		Straight-Circle-Straight generation mode(using reflect).

		Parameters:
			x (float): x of goal position
			y (float): y of goal position
			phi (float): goal orientation

		Returns:
			paths (list): Available paths
		"""
		paths = []

		flag, t, u, v = self.SLS(x, y, phi)
		if flag:
			paths.append(self.Path(lengths=[t, u, v], ctypes=["S", "L", "S"]))

		flag, t, u, v = self.SLS(x, -y, -phi)
		if flag:
			paths.append(self.Path(lengths=[t, u, v], ctypes=["S", "R", "S"]))

		return paths

	def CCC(self, x: float, y: float, phi: float):
		"""
		# 8
		Circle-Circle-Circle generation mode(using reflect, timeflip and backwards).

		Parameters:
			x (float): x of goal position
			y (float): y of goal position
			phi (float): goal orientation

		Returns:
			paths (list): Available paths
		"""
		paths = []

		# L+R-L-
		flag, t, u, v = self.LRL(x, y, phi)
		if flag:
			paths.append(self.Path(lengths=[t, u, v], ctypes=["L", "R", "L"]))

		# timefilp: L-R+L+
		flag, t, u, v = self.LRL(-x, y, -phi)
		if flag:
			paths.append(self.Path(lengths=[-t, -u, -v], ctypes=["L", "R", "L"]))

		# reflect: R+L-R-
		flag, t, u, v = self.LRL(x, -y, -phi)
		if flag:
			paths.append(self.Path(lengths=[t, u, v], ctypes=["R", "L", "R"]))

		# timeflip + reflect: R-L+R+
		flag, t, u, v = self.LRL(-x, -y, phi)
		if flag:
			paths.append(self.Path(lengths=[-t, -u, -v], ctypes=["R", "L", "R"]))

		# backwards
		xb = x * math.cos(phi) + y * math.sin(phi)
		yb = x * math.sin(phi) - y * math.cos(phi)

		# backwards: L-R-L+
		flag, t, u, v = self.LRL(xb, yb, phi)
		if flag:
			paths.append(self.Path(lengths=[v, u, t], ctypes=["L", "R", "L"]))

		# backwards + timefilp: L+R+L-
		flag, t, u, v = self.LRL(-xb, yb, -phi)
		if flag:
			paths.append(self.Path(lengths=[-v, -u, -t], ctypes=["L", "R", "L"]))

		# backwards + reflect: R-L-R+
		flag, t, u, v = self.LRL(xb, -yb, -phi)
		if flag:
			paths.append(self.Path(lengths=[v, u, t], ctypes=["R", "L", "R"]))

		# backwards + timeflip + reflect: R+L+R-
		flag, t, u, v = self.LRL(-xb, -yb, phi)
		if flag:
			paths.append(self.Path(lengths=[-v, -u, -t], ctypes=["R", "L", "R"]))

		return paths

	def CSC(self, x: float, y: float, phi: float):
		"""
		# 8
		Circle-Straight-Circle generation mode(using reflect, timeflip and backwards).

		Parameters:
			x (float): x of goal position
			y (float): y of goal position
			phi (float): goal orientation

		Returns:
			paths (list): Available paths
		"""
		paths = []

		# L+S+L+
		flag, t, u, v = self.LSL(x, y, phi)
		if flag:
			paths.append(self.Path(lengths=[t, u, v], ctypes=["L", "S", "L"]))

		# timefilp: L-S-L-
		flag, t, u, v = self.LSL(-x, y, -phi)
		if flag:
			paths.append(self.Path(lengths=[-t, -u, -v], ctypes=["L", "S", "L"]))

		# reflect: R+S+R+
		flag, t, u, v = self.LSL(x, -y, -phi)
		if flag:
			paths.append(self.Path(lengths=[t, u, v], ctypes=["R", "S", "R"]))

		# timeflip + reflect: R-S-R-
		flag, t, u, v = self.LSL(-x, -y, phi)
		if flag:
			paths.append(self.Path(lengths=[-t, -u, -v], ctypes=["R", "S", "R"]))

		# L+S+R+
		flag, t, u, v = self.LSR(x, y, phi)
		if flag:
			paths.append(self.Path(lengths=[t, u, v], ctypes=["L", "S", "R"]))

		# timefilp: L-S-R-
		flag, t, u, v = self.LSR(-x, y, -phi)
		if flag:
			paths.append(self.Path(lengths=[-t, -u, -v], ctypes=["L", "S", "R"]))

		# reflect: R+S+L+
		flag, t, u, v = self.LSR(x, -y, -phi)
		if flag:
			paths.append(self.Path(lengths=[t, u, v], ctypes=["R", "S", "L"]))

		# timeflip + reflect: R+S+l-
		flag, t, u, v = self.LSR(-x, -y, phi)
		if flag:
			paths.append(self.Path(lengths=[-t, -u, -v], ctypes=["R", "S", "L"]))

		return paths

	def CCCC(self, x: float, y: float, phi: float):
		"""
		# 8
		Circle-Circle(beta)-Circle(beta)-Circle generation mode
		(using reflect, timeflip and backwards).

		Parameters:
			x (float): x of goal position
			y (float): y of goal position
			phi (float): goal orientation

		Returns:
			paths (list): Available paths
		"""
		paths = []

		# L+R+L-R-
		flag, t, u, v = self.LRLRn(x, y, phi)
		if flag:
			paths.append(self.Path(lengths=[t, u, -u, v], ctypes=["L", "R", "L", "R"]))

		# timefilp: L-R-L+R+
		flag, t, u, v = self.LRLRn(-x, y, -phi)
		if flag:
			paths.append(self.Path(lengths=[-t, -u, u, -v], ctypes=["L", "R", "L", "R"]))
		
		# reflect: R+L+R-L-
		flag, t, u, v = self.LRLRn(x, -y, -phi)
		if flag:
			paths.append(self.Path(lengths=[t, u, -u, v], ctypes=["R", "L", "R", "L"]))

		# timeflip + reflect: R-L-R+L+
		flag, t, u, v = self.LRLRn(-x, -y, phi)
		if flag:
			paths.append(self.Path(lengths=[-t, -u, u, -v], ctypes=["R", "L", "R", "L"]))

		# L+R-L-R+
		flag, t, u, v = self.LRLRp(x, y, phi)
		if flag:
			paths.append(self.Path(lengths=[t, u, u, v], ctypes=["L", "R", "L", "R"]))

		# timefilp: L-R+L+R-
		flag, t, u, v = self.LRLRp(-x, y, -phi)
		if flag:
			paths.append(self.Path(lengths=[-t, -u, -u, -v], ctypes=["L", "R", "L", "R"]))

		# reflect: R+L-R-L+
		flag, t, u, v = self.LRLRp(x, -y, -phi)
		if flag:
			paths.append(self.Path(lengths=[t, u, u, v], ctypes=["R", "L", "R", "L"]))

		# timeflip + reflect: R-L+R+L-
		flag, t, u, v = self.LRLRp(-x, -y, phi)
		if flag:
			paths.append(self.Path(lengths=[-t, -u, -u, -v], ctypes=["R", "L", "R", "L"]))

		return paths

	def CCSC(self, x: float, y: float, phi: float):
		"""
		# 16
		Circle-Circle(pi/2)-Straight-Circle and Circle-Straight-Circle(pi/2)-Circle
		generation mode (using reflect, timeflip and backwards).

		Parameters:
			x (float): x of goal position
			y (float): y of goal position
			phi (float): goal orientation

		Returns:
			paths (list): Available paths
		"""
		paths = []

		# L+R-(pi/2)S-L-
		flag, t, u, v = self.LRSL(x, y, phi)
		if flag:
			paths.append(self.Path(lengths=[t, -0.5 * math.pi, u, v], ctypes=["L", "R", "S", "L"]))

		# timefilp: L-R+(pi/2)S+L+
		flag, t, u, v = self.LRSL(-x, y, -phi)
		if flag:
			paths.append(self.Path(lengths=[-t, 0.5 * math.pi, -u, -v], ctypes=["L", "R", "S", "L"]))

		# reflect: R+L-(pi/2)S-R-
		flag, t, u, v = self.LRSL(x, -y, -phi)
		if flag:
			paths.append(self.Path(lengths=[t, -0.5 * math.pi, u, v], ctypes=["R", "L", "S", "R"]))

		# timeflip + reflect: R-L+(pi/2)S+R+
		flag, t, u, v = self.LRSL(-x, -y, phi)
		if flag:
			paths.append(self.Path(lengths=[-t, 0.5 * math.pi, -u, -v], ctypes=["R", "L", "S", "R"]))

		# L+R-(pi/2)S-R-
		flag, t, u, v = self.LRSR(x, y, phi)
		if flag:
			paths.append(self.Path(lengths=[t, -0.5 * math.pi, u, v], ctypes=["L", "R", "S", "R"]))

		# timefilp: L-R+(pi/2)S+R+
		flag, t, u, v = self.LRSR(-x, y, -phi)
		if flag:
			paths.append(self.Path(lengths=[-t, 0.5 * math.pi, -u, -v], ctypes=["L", "R", "S", "R"]))

		# reflect: R+L-(pi/2)S-L-
		flag, t, u, v = self.LRSR(x, -y, -phi)
		if flag:
			paths.append(self.Path(lengths=[t, -0.5 * math.pi, u, v], ctypes=["R", "L", "S", "L"]))

		# timeflip + reflect: R-L+(pi/2)S+L+
		flag, t, u, v = self.LRSR(-x, -y, phi)
		if flag:
			paths.append(self.Path(lengths=[-t, 0.5 * math.pi, -u, -v], ctypes=["R", "L", "S", "L"]))

		# backwards
		xb = x * math.cos(phi) + y * math.sin(phi)
		yb = x * math.sin(phi) - y * math.cos(phi)

		# backwards: L-S-R-(pi/2)L+
		flag, t, u, v = self.LRSL(xb, yb, phi)
		if flag:
			paths.append(self.Path(lengths=[v, u, -0.5 * math.pi, t], ctypes=["L", "S", "R", "L"]))

		# backwards + timefilp: L+S+R+(pi/2)L-
		flag, t, u, v = self.LRSL(-xb, yb, -phi)
		if flag:
			paths.append(self.Path(lengths=[-v, -u, 0.5 * math.pi, -t], ctypes=["L", "S", "R", "L"]))

		# backwards + reflect: R-S-L-(pi/2)R+
		flag, t, u, v = self.LRSL(xb, -yb, -phi)
		if flag:
			paths.append(self.Path(lengths=[v, u, -0.5 * math.pi, t], ctypes=["R", "S", "L", "R"]))

		# backwards + timefilp + reflect: R+S+L+(pi/2)R-
		flag, t, u, v = self.LRSL(-xb, -yb, phi)
		if flag:
			paths.append(self.Path(lengths=[-v, -u, 0.5 * math.pi, -t], ctypes=["R", "S", "L", "R"]))

		# backwards: R-S-R-(pi/2)L+
		flag, t, u, v = self.LRSR(xb, yb, phi)
		if flag:
			paths.append(self.Path(lengths=[v, u, -0.5 * math.pi, t], ctypes=["R", "S", "R", "L"]))

		# backwards + timefilp: R+S+R+(pi/2)L-
		flag, t, u, v = self.LRSR(-xb, yb, -phi)
		if flag:
			paths.append(self.Path(lengths=[-v, -u, 0.5 * math.pi, -t], ctypes=["R", "S", "R", "L"]))

		# backwards + reflect: L-S-L-(pi/2)R+
		flag, t, u, v = self.LRSR(xb, -yb, -phi)
		if flag:
			paths.append(self.Path(lengths=[v, u, -0.5 * math.pi, t], ctypes=["L", "S", "L", "R"]))

		# backwards + timefilp + reflect: L+S+L+(pi/2)R-
		flag, t, u, v = self.LRSR(-xb, -yb, phi)
		if flag:
			paths.append(self.Path(lengths=[-v, -u, 0.5 * math.pi, -t], ctypes=["L", "S", "L", "R"]))

		return paths

	def CCSCC(self, x: float, y: float, phi: float):
		"""
		# 4
		Circle-Circle(pi/2)-Straight--Circle(pi/2)-Circle generation mode (using reflect, timeflip and backwards).

		Parameters:
			x (float): x of goal position
			y (float): y of goal position
			phi (float): goal orientation

		Returns:
			paths (list): Available paths
		"""
		paths = []

		# L+R-(pi/2)S-L-(pi/2)R+
		flag, t, u, v = self.LRSLR(x, y, phi)
		if flag:
			paths.append(self.Path(lengths=[t, -0.5 * math.pi, u, -0.5 * math.pi, v], ctypes=["L", "R", "S", "L", "R"]))

		# timefilp: L-R+(pi/2)S+L+(pi/2)R-
		flag, t, u, v = self.LRSLR(-x, y, -phi)
		if flag:
			paths.append(self.Path(lengths=[-t, 0.5 * math.pi, -u, 0.5 * math.pi, -v], ctypes=["L", "R", "S", "L", "R"]))

		# reflect: R+L-(pi/2)S-R-(pi/2)L+
		flag, t, u, v = self.LRSLR(x, -y, -phi)
		if flag:
			paths.append(self.Path(lengths=[t, -0.5 * math.pi, u, -0.5 * math.pi, v], ctypes=["R", "L", "S", "R", "L"]))

		# timefilp + reflect: R-L+(pi/2)S+R+(pi/2)L-
		flag, t, u, v = self.LRSLR(-x, -y, phi)
		if flag:
			paths.append(self.Path(lengths=[-t, 0.5 * math.pi, -u, 0.5 * math.pi, -v], ctypes=["R", "L", "S", "R", "L"]))

		return paths

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
			new_x = x + length / self.max_curv * math.cos(yaw)
			new_y = y + length / self.max_curv * math.sin(yaw)
			new_yaw = yaw
		elif mode == "L":
			new_x = x + (math.sin(yaw + length) - math.sin(yaw)) / self.max_curv
			new_y = y - (math.cos(yaw + length) - math.cos(yaw)) / self.max_curv
			new_yaw = yaw + length
		elif mode == "R":
			new_x = x - (math.sin(yaw - length) - math.sin(yaw)) / self.max_curv
			new_y = y + (math.cos(yaw - length) - math.cos(yaw)) / self.max_curv
			new_yaw = yaw - length
		else:
			raise NotImplementedError

		return new_x, new_y, new_yaw

	def generation(self, start_pose: tuple, goal_pose: tuple):
		"""
		Generate the Reeds Shepp Curve.

		Parameters:
			start_pose (tuple): Initial pose (x, y, yaw)
			goal_pose (tuple): Target pose (x, y, yaw)

		Returns:
			best_cost (float): Best planning path length
			best_mode: Best motion modes
			x_list (list): Trajectory of x
			y_list (list): Trajectory of y
			yaw_list (list): Trajectory of yaw
		"""
		sx, sy, syaw = start_pose
		gx, gy, gyaw = goal_pose

		# coordinate transformation
		dx, dy, dyaw = gx - sx, gy - sy, gyaw - syaw
		x = (math.cos(syaw) * dx + math.sin(syaw) * dy) * self.max_curv
		y = (-math.sin(syaw) * dx + math.cos(syaw) * dy) * self.max_curv

		# select the best motion
		planners = [self.SCS, self.CCC, self.CSC, self.CCCC, self.CCSC, self.CCSCC]
		best_path, best_cost = None, float("inf")

		for planner in planners:
			paths = planner(x, y, dyaw)
			for path in paths:
				if path.path_length < best_cost:
					best_path, best_cost = path, path.path_length

		# interpolation
		points_num = int(best_cost / self.step) + len(best_path.lengths) + 3
		x_list = [0.0 for _ in range(points_num)]
		y_list = [0.0 for _ in range(points_num)]
		yaw_list = [0.0 for _ in range(points_num)]

		i = 0
		for mode_, seg_length in zip(best_path.ctypes, best_path.lengths):
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
		x_list_ = [math.cos(-syaw) * ix + math.sin(-syaw) * iy + sx for (ix, iy) in zip(x_list, y_list)]
		y_list_ = [-math.sin(-syaw) * ix + math.cos(-syaw) * iy + sy for (ix, iy) in zip(x_list, y_list)]
		yaw_list_ = [self.pi2pi(iyaw + syaw) for iyaw in yaw_list]

		return best_cost / self.max_curv, best_path.ctypes, x_list_, y_list_, yaw_list_

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
		# # static
		# plt.plot(path_x, path_y, linewidth=2, c="#1f77b4")
		# for x, y, theta in points:
		# 	Plot.plotArrow(x, y, np.deg2rad(theta), 2, 'blueviolet')

		# dynamic
		plt.ion()
		for i in range(len(path_x)):
			plt.clf()
			plt.gcf().canvas.mpl_connect('key_release_event',
											lambda event: [exit(0) if event.key == 'escape' else None])
			plt.plot(path_x, path_y, linewidth=2, c="#1f77b4")
			for x, y, theta in points:
				Plot.plotArrow(x, y, np.deg2rad(theta), 2, 'blueviolet')
			Plot.plotCar(path_x[i], path_y[i], path_yaw[i], 1.5, 3, "black")
			plt.axis("equal")
			plt.title(str(self))
			plt.draw()
			plt.pause(0.001)

		plt.axis("equal")
		plt.title(str(self))
		plt.show()


	def _calTauOmega(self, u, v, xi, eta, phi):
		delta = self.M(u - v)
		A = math.sin(u) - math.sin(delta)
		B = math.cos(u) - math.cos(delta) - 1.0

		t1 = math.atan2(eta * A - xi * B, xi * A + eta * B)
		t2 = 2.0 * (math.cos(delta) - math.cos(v) - math.cos(u)) + 3.0

		tau = self.M(t1 + math.pi) if t2 < 0 else self.M(t1)
		omega = self.M(tau - u + v - phi)

		return tau, omega
