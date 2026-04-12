"""
@file: reeds_shepp.py
@author: Yang Haodong, Wu Maojia
@update: 2026.4.12
"""
from typing import List, Tuple, Dict, Any
import math

from python_motion_planning.traj_optimizer.base_curve_generator import BaseCurveGenerator
from python_motion_planning.common.utils.geometry import Geometry


class ReedsShepp(BaseCurveGenerator):
    """
    Class for Reeds-Shepp curve generator.

    Args:
        *args: see the parent class.
        max_curv: The maximum curvature of the curve.
        *args: see the parent class.

    References:
        [1] Optimal paths for a car that goes both forwards and backwards

    Examples:
        >>> import math
        >>> generator = ReedsShepp(step=0.1, max_curv=1.0)
        >>> points = [(0.0, 0.0, 0.0), (10.0, 10.0, -math.pi/2), (20.0, 5.0, math.pi/3)]
        >>> path, curve_info = generator.generate(points)
        >>> print(curve_info['success'])
        True
    """
    def __init__(self, *args, 
                 max_curv: float = 1.0,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_curv = max_curv

    def __str__(self) -> str:
        return "Reeds Shepp Curve"

    class _Path:
        """
        Container for a single Reeds-Shepp path candidate.
        """
        def __init__(self, lengths: List[float] = None, ctypes: List[str] = None):
            self.lengths = lengths if lengths is not None else []
            self.ctypes = ctypes if ctypes is not None else []
            self.path_length = sum(abs(i) for i in self.lengths)

    def generate(self, points: List[Tuple[float, float, float]]) -> Tuple[List[Tuple[float, float, float]], Dict[str, Any]]:
        """
        Generate a concatenated Reeds-Shepp curve through a list of poses.

        Args:
            points: A list of poses (x, y, yaw) in world frame.

        Returns:
            path: A list of (x, y, yaw) waypoints of the generated curve in world frame.
            curve_info: A dictionary containing the curve information (success, length).
        """
        if len(points) < 2:
            return [], {"success": False, "length": 0.0}

        path: List[Tuple[float, float, float]] = []
        total_length = 0.0
        for i in range(len(points) - 1):
            best_cost, _, x_list, y_list, yaw_list = self._generate_segment(
                points[i], points[i + 1])
            if best_cost is None:
                return [], {"success": False, "length": 0.0}
            total_length += best_cost
            
            start = 1 if i > 0 else 0
            for x, y, yaw in zip(x_list[start:], y_list[start:], yaw_list[start:]):
                path.append((float(x), float(y), float(yaw)))

        total_length = float(total_length)

        return path, {"success": True, "length": total_length}

    def _r(self, x: float, y: float) -> Tuple[float, float]:
        """
        Convert (x, y) to polar coordinates (r, theta).

        Args:
            x: x-coordinate value.
            y: y-coordinate value.

        Returns:
            r, theta: Polar coordinates.
        """
        return math.hypot(x, y), math.atan2(y, x)

    def _sls(self, x: float, y: float, phi: float):
        """
        Straight-Left-Straight generation mode.
        """
        phi = Geometry.regularize_orient(phi)

        if y > 0.0 and 0.0 < phi < math.pi * 0.99:
            xd = -y / math.tan(phi) + x
            t = xd - math.tan(phi / 2.0)
            u = phi
            v = math.sqrt((x - xd) ** 2 + y ** 2) - math.tan(phi / 2.0)
            return True, t, u, v
        if y < 0.0 and 0.0 < phi < math.pi * 0.99:
            xd = -y / math.tan(phi) + x
            t = xd - math.tan(phi / 2.0)
            u = phi
            v = -math.sqrt((x - xd) ** 2 + y ** 2) - math.tan(phi / 2.0)
            return True, t, u, v
        return False, 0.0, 0.0, 0.0

    def _lrl(self, x: float, y: float, phi: float):
        """
        Left-Right-Left generation mode (L+R-L-).
        """
        r, theta = self._r(x - math.sin(phi), y - 1.0 + math.cos(phi))

        if r <= 4.0:
            u = -2.0 * math.asin(0.25 * r)
            t = Geometry.regularize_orient(theta + 0.5 * u + math.pi)
            v = Geometry.regularize_orient(phi - t + u)
            if t >= 0.0 and u <= 0.0:
                return True, t, u, v
        return False, 0.0, 0.0, 0.0

    def _lsl(self, x: float, y: float, phi: float):
        """
        Left-Straight-Left generation mode (L+S+L+).
        """
        u, t = self._r(x - math.sin(phi), y - 1.0 + math.cos(phi))
        if t >= 0.0:
            v = Geometry.regularize_orient(phi - t)
            if v >= 0.0:
                return True, t, u, v
        return False, 0.0, 0.0, 0.0

    def _lsr(self, x: float, y: float, phi: float):
        """
        Left-Straight-Right generation mode (L+S+R+).
        """
        r, theta = self._r(x + math.sin(phi), y - 1.0 - math.cos(phi))
        r = r ** 2
        if r >= 4.0:
            u = math.sqrt(r - 4.0)
            t = Geometry.regularize_orient(theta + math.atan2(2.0, u))
            v = Geometry.regularize_orient(t - phi)
            if t >= 0.0 and v >= 0.0:
                return True, t, u, v
        return False, 0.0, 0.0, 0.0

    def _lrlrn(self, x: float, y: float, phi: float):
        """
        Left-Right(beta)-Left(beta)-Right generation mode (L+R+L-R-).
        """
        xi = x + math.sin(phi)
        eta = y - 1.0 - math.cos(phi)
        rho = 0.25 * (2.0 + math.sqrt(xi * xi + eta * eta))

        if rho <= 1.0:
            u = math.acos(rho)
            t, v = self._cal_tau_omega(u, -u, xi, eta, phi)
            if t >= 0.0 and v <= 0.0:
                return True, t, u, v
        return False, 0.0, 0.0, 0.0

    def _lrlrp(self, x: float, y: float, phi: float):
        """
        Left-Right(beta)-Left(beta)-Right generation mode (L+R-L-R+).
        """
        xi = x + math.sin(phi)
        eta = y - 1.0 - math.cos(phi)
        rho = (20.0 - xi * xi - eta * eta) / 16.0

        if 0.0 <= rho <= 1.0:
            u = -math.acos(rho)
            if u >= -0.5 * math.pi:
                t, v = self._cal_tau_omega(u, u, xi, eta, phi)
                if t >= 0.0 and v >= 0.0:
                    return True, t, u, v
        return False, 0.0, 0.0, 0.0

    def _lrsr(self, x: float, y: float, phi: float):
        """
        Left-Right(pi/2)-Straight-Right generation mode (L+R-S-R-).
        """
        xi = x + math.sin(phi)
        eta = y - 1.0 - math.cos(phi)
        rho, theta = self._r(-eta, xi)

        if rho >= 2.0:
            t = theta
            u = 2.0 - rho
            v = Geometry.regularize_orient(t + 0.5 * math.pi - phi)
            if t >= 0.0 and u <= 0.0 and v <= 0.0:
                return True, t, u, v
        return False, 0.0, 0.0, 0.0

    def _lrsl(self, x: float, y: float, phi: float):
        """
        Left-Right(pi/2)-Straight-Left generation mode (L+R-S-L-).
        """
        xi = x - math.sin(phi)
        eta = y - 1.0 + math.cos(phi)
        rho, theta = self._r(xi, eta)

        if rho >= 2.0:
            r = math.sqrt(rho * rho - 4.0)
            u = 2.0 - r
            t = Geometry.regularize_orient(theta + math.atan2(r, -2.0))
            v = Geometry.regularize_orient(phi - 0.5 * math.pi - t)
            if t >= 0.0 and u <= 0.0 and v <= 0.0:
                return True, t, u, v
        return False, 0.0, 0.0, 0.0

    def _lrslr(self, x: float, y: float, phi: float):
        """
        Left-Right(pi/2)-Straight-Left(pi/2)-Right generation mode (L+R-S-L-R+).
        """
        xi = x + math.sin(phi)
        eta = y - 1.0 - math.cos(phi)
        r, _ = self._r(xi, eta)

        if r >= 2.0:
            u = 4.0 - math.sqrt(r * r - 4.0)
            if u <= 0.0:
                t = Geometry.regularize_orient(math.atan2((4.0 - u) * xi - 2.0 * eta, -2.0 * xi + (u - 4.0) * eta))
                v = Geometry.regularize_orient(t - phi)
                if t >= 0.0 and v >= 0.0:
                    return True, t, u, v
        return False, 0.0, 0.0, 0.0

    def _scs(self, x: float, y: float, phi: float) -> List["_Path"]:
        """
        Straight-Circle-Straight generation mode family (using reflect).
        """
        paths = []

        flag, t, u, v = self._sls(x, y, phi)
        if flag:
            paths.append(self._Path(lengths=[t, u, v], ctypes=["S", "L", "S"]))

        flag, t, u, v = self._sls(x, -y, -phi)
        if flag:
            paths.append(self._Path(lengths=[t, u, v], ctypes=["S", "R", "S"]))

        return paths

    def _ccc(self, x: float, y: float, phi: float) -> List["_Path"]:
        """
        Circle-Circle-Circle generation mode family (using reflect, timeflip and backwards).
        """
        paths = []

        flag, t, u, v = self._lrl(x, y, phi)
        if flag:
            paths.append(self._Path(lengths=[t, u, v], ctypes=["L", "R", "L"]))

        flag, t, u, v = self._lrl(-x, y, -phi)
        if flag:
            paths.append(self._Path(lengths=[-t, -u, -v], ctypes=["L", "R", "L"]))

        flag, t, u, v = self._lrl(x, -y, -phi)
        if flag:
            paths.append(self._Path(lengths=[t, u, v], ctypes=["R", "L", "R"]))

        flag, t, u, v = self._lrl(-x, -y, phi)
        if flag:
            paths.append(self._Path(lengths=[-t, -u, -v], ctypes=["R", "L", "R"]))

        xb = x * math.cos(phi) + y * math.sin(phi)
        yb = x * math.sin(phi) - y * math.cos(phi)

        flag, t, u, v = self._lrl(xb, yb, phi)
        if flag:
            paths.append(self._Path(lengths=[v, u, t], ctypes=["L", "R", "L"]))

        flag, t, u, v = self._lrl(-xb, yb, -phi)
        if flag:
            paths.append(self._Path(lengths=[-v, -u, -t], ctypes=["L", "R", "L"]))

        flag, t, u, v = self._lrl(xb, -yb, -phi)
        if flag:
            paths.append(self._Path(lengths=[v, u, t], ctypes=["R", "L", "R"]))

        flag, t, u, v = self._lrl(-xb, -yb, phi)
        if flag:
            paths.append(self._Path(lengths=[-v, -u, -t], ctypes=["R", "L", "R"]))

        return paths

    def _csc(self, x: float, y: float, phi: float) -> List["_Path"]:
        """
        Circle-Straight-Circle generation mode family (using reflect, timeflip and backwards).
        """
        paths = []

        flag, t, u, v = self._lsl(x, y, phi)
        if flag:
            paths.append(self._Path(lengths=[t, u, v], ctypes=["L", "S", "L"]))

        flag, t, u, v = self._lsl(-x, y, -phi)
        if flag:
            paths.append(self._Path(lengths=[-t, -u, -v], ctypes=["L", "S", "L"]))

        flag, t, u, v = self._lsl(x, -y, -phi)
        if flag:
            paths.append(self._Path(lengths=[t, u, v], ctypes=["R", "S", "R"]))

        flag, t, u, v = self._lsl(-x, -y, phi)
        if flag:
            paths.append(self._Path(lengths=[-t, -u, -v], ctypes=["R", "S", "R"]))

        flag, t, u, v = self._lsr(x, y, phi)
        if flag:
            paths.append(self._Path(lengths=[t, u, v], ctypes=["L", "S", "R"]))

        flag, t, u, v = self._lsr(-x, y, -phi)
        if flag:
            paths.append(self._Path(lengths=[-t, -u, -v], ctypes=["L", "S", "R"]))

        flag, t, u, v = self._lsr(x, -y, -phi)
        if flag:
            paths.append(self._Path(lengths=[t, u, v], ctypes=["R", "S", "L"]))

        flag, t, u, v = self._lsr(-x, -y, phi)
        if flag:
            paths.append(self._Path(lengths=[-t, -u, -v], ctypes=["R", "S", "L"]))

        return paths

    def _cccc(self, x: float, y: float, phi: float) -> List["_Path"]:
        """
        Circle-Circle(beta)-Circle(beta)-Circle generation mode family
        (using reflect, timeflip and backwards).
        """
        paths = []

        flag, t, u, v = self._lrlrn(x, y, phi)
        if flag:
            paths.append(self._Path(lengths=[t, u, -u, v], ctypes=["L", "R", "L", "R"]))

        flag, t, u, v = self._lrlrn(-x, y, -phi)
        if flag:
            paths.append(self._Path(lengths=[-t, -u, u, -v], ctypes=["L", "R", "L", "R"]))

        flag, t, u, v = self._lrlrn(x, -y, -phi)
        if flag:
            paths.append(self._Path(lengths=[t, u, -u, v], ctypes=["R", "L", "R", "L"]))

        flag, t, u, v = self._lrlrn(-x, -y, phi)
        if flag:
            paths.append(self._Path(lengths=[-t, -u, u, -v], ctypes=["R", "L", "R", "L"]))

        flag, t, u, v = self._lrlrp(x, y, phi)
        if flag:
            paths.append(self._Path(lengths=[t, u, u, v], ctypes=["L", "R", "L", "R"]))

        flag, t, u, v = self._lrlrp(-x, y, -phi)
        if flag:
            paths.append(self._Path(lengths=[-t, -u, -u, -v], ctypes=["L", "R", "L", "R"]))

        flag, t, u, v = self._lrlrp(x, -y, -phi)
        if flag:
            paths.append(self._Path(lengths=[t, u, u, v], ctypes=["R", "L", "R", "L"]))

        flag, t, u, v = self._lrlrp(-x, -y, phi)
        if flag:
            paths.append(self._Path(lengths=[-t, -u, -u, -v], ctypes=["R", "L", "R", "L"]))

        return paths

    def _ccsc(self, x: float, y: float, phi: float) -> List["_Path"]:
        """
        Circle-Circle(pi/2)-Straight-Circle and Circle-Straight-Circle(pi/2)-Circle
        generation mode family (using reflect, timeflip and backwards).
        """
        paths = []

        flag, t, u, v = self._lrsl(x, y, phi)
        if flag:
            paths.append(self._Path(lengths=[t, -0.5 * math.pi, u, v], ctypes=["L", "R", "S", "L"]))

        flag, t, u, v = self._lrsl(-x, y, -phi)
        if flag:
            paths.append(self._Path(lengths=[-t, 0.5 * math.pi, -u, -v], ctypes=["L", "R", "S", "L"]))

        flag, t, u, v = self._lrsl(x, -y, -phi)
        if flag:
            paths.append(self._Path(lengths=[t, -0.5 * math.pi, u, v], ctypes=["R", "L", "S", "R"]))

        flag, t, u, v = self._lrsl(-x, -y, phi)
        if flag:
            paths.append(self._Path(lengths=[-t, 0.5 * math.pi, -u, -v], ctypes=["R", "L", "S", "R"]))

        flag, t, u, v = self._lrsr(x, y, phi)
        if flag:
            paths.append(self._Path(lengths=[t, -0.5 * math.pi, u, v], ctypes=["L", "R", "S", "R"]))

        flag, t, u, v = self._lrsr(-x, y, -phi)
        if flag:
            paths.append(self._Path(lengths=[-t, 0.5 * math.pi, -u, -v], ctypes=["L", "R", "S", "R"]))

        flag, t, u, v = self._lrsr(x, -y, -phi)
        if flag:
            paths.append(self._Path(lengths=[t, -0.5 * math.pi, u, v], ctypes=["R", "L", "S", "L"]))

        flag, t, u, v = self._lrsr(-x, -y, phi)
        if flag:
            paths.append(self._Path(lengths=[-t, 0.5 * math.pi, -u, -v], ctypes=["R", "L", "S", "L"]))

        xb = x * math.cos(phi) + y * math.sin(phi)
        yb = x * math.sin(phi) - y * math.cos(phi)

        flag, t, u, v = self._lrsl(xb, yb, phi)
        if flag:
            paths.append(self._Path(lengths=[v, u, -0.5 * math.pi, t], ctypes=["L", "S", "R", "L"]))

        flag, t, u, v = self._lrsl(-xb, yb, -phi)
        if flag:
            paths.append(self._Path(lengths=[-v, -u, 0.5 * math.pi, -t], ctypes=["L", "S", "R", "L"]))

        flag, t, u, v = self._lrsl(xb, -yb, -phi)
        if flag:
            paths.append(self._Path(lengths=[v, u, -0.5 * math.pi, t], ctypes=["R", "S", "L", "R"]))

        flag, t, u, v = self._lrsl(-xb, -yb, phi)
        if flag:
            paths.append(self._Path(lengths=[-v, -u, 0.5 * math.pi, -t], ctypes=["R", "S", "L", "R"]))

        flag, t, u, v = self._lrsr(xb, yb, phi)
        if flag:
            paths.append(self._Path(lengths=[v, u, -0.5 * math.pi, t], ctypes=["R", "S", "R", "L"]))

        flag, t, u, v = self._lrsr(-xb, yb, -phi)
        if flag:
            paths.append(self._Path(lengths=[-v, -u, 0.5 * math.pi, -t], ctypes=["R", "S", "R", "L"]))

        flag, t, u, v = self._lrsr(xb, -yb, -phi)
        if flag:
            paths.append(self._Path(lengths=[v, u, -0.5 * math.pi, t], ctypes=["L", "S", "L", "R"]))

        flag, t, u, v = self._lrsr(-xb, -yb, phi)
        if flag:
            paths.append(self._Path(lengths=[-v, -u, 0.5 * math.pi, -t], ctypes=["L", "S", "L", "R"]))

        return paths

    def _ccscc(self, x: float, y: float, phi: float) -> List["_Path"]:
        """
        Circle-Circle(pi/2)-Straight-Circle(pi/2)-Circle generation mode family
        (using reflect, timeflip and backwards).
        """
        paths = []

        flag, t, u, v = self._lrslr(x, y, phi)
        if flag:
            paths.append(self._Path(lengths=[t, -0.5 * math.pi, u, -0.5 * math.pi, v],
                                    ctypes=["L", "R", "S", "L", "R"]))

        flag, t, u, v = self._lrslr(-x, y, -phi)
        if flag:
            paths.append(self._Path(lengths=[-t, 0.5 * math.pi, -u, 0.5 * math.pi, -v],
                                    ctypes=["L", "R", "S", "L", "R"]))

        flag, t, u, v = self._lrslr(x, -y, -phi)
        if flag:
            paths.append(self._Path(lengths=[t, -0.5 * math.pi, u, -0.5 * math.pi, v],
                                    ctypes=["R", "L", "S", "R", "L"]))

        flag, t, u, v = self._lrslr(-x, -y, phi)
        if flag:
            paths.append(self._Path(lengths=[-t, 0.5 * math.pi, -u, 0.5 * math.pi, -v],
                                    ctypes=["R", "L", "S", "R", "L"]))

        return paths

    def _interpolate(self, mode: str, length: float,
                     init_pose: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Planning path interpolation.

        Args:
            mode: Motion type, one of {"L", "S", "R"}.
            length: Single step motion path length.
            init_pose: Initial pose (x, y, yaw).

        Returns:
            new_pose: New pose (new_x, new_y, new_yaw) after moving.
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

    def _generate_segment(self, start_pose: Tuple[float, float, float],
                          goal_pose: Tuple[float, float, float]):
        """
        Generate a single Reeds-Shepp curve segment between two poses.

        Args:
            start_pose: Initial pose (x, y, yaw).
            goal_pose: Target pose (x, y, yaw).

        Returns:
            best_cost: Best planning path length in world units.
            best_mode: Best motion modes.
            x_list: Trajectory of x.
            y_list: Trajectory of y.
            yaw_list: Trajectory of yaw.
        """
        sx, sy, syaw = start_pose
        gx, gy, gyaw = goal_pose

        dx, dy, dyaw = gx - sx, gy - sy, gyaw - syaw
        x = (math.cos(syaw) * dx + math.sin(syaw) * dy) * self.max_curv
        y = (-math.sin(syaw) * dx + math.cos(syaw) * dy) * self.max_curv

        planners = [self._scs, self._ccc, self._csc, self._cccc, self._ccsc, self._ccscc]
        best_path, best_cost = None, float("inf")

        for planner in planners:
            paths = planner(x, y, dyaw)
            for path in paths:
                if path.path_length < best_cost:
                    best_path, best_cost = path, path.path_length

        if best_path is None:
            return None, None, [], [], []

        points_num = int(best_cost / self.step) + len(best_path.lengths) + 3
        x_list = [0.0 for _ in range(points_num)]
        y_list = [0.0 for _ in range(points_num)]
        yaw_list = [0.0 for _ in range(points_num)]

        i = 0
        for mode_, seg_length in zip(best_path.ctypes, best_path.lengths):
            d_length = self.step if seg_length > 0.0 else -self.step
            current_x, current_y, current_yaw = x_list[i], y_list[i], yaw_list[i]
            length = d_length
            while abs(length) <= abs(seg_length):
                i += 1
                current_x, current_y, current_yaw = self._interpolate(
                    mode_, d_length, (current_x, current_y, current_yaw)
                )
                x_list[i], y_list[i], yaw_list[i] = current_x, current_y, current_yaw
                length += d_length
            
            i += 1
            remainder = seg_length - (length - d_length)
            x_list[i], y_list[i], yaw_list[i] = self._interpolate(
                mode_, remainder, (x_list[i-1], y_list[i-1], yaw_list[i-1])
            )

        if len(x_list) <= 1:
            return None, None, [], [], []

        while len(x_list) >= 1 and x_list[-1] == 0.0:
            x_list.pop()
            y_list.pop()
            yaw_list.pop()

        x_list_ = [math.cos(-syaw) * ix + math.sin(-syaw) * iy + sx for (ix, iy) in zip(x_list, y_list)]
        y_list_ = [-math.sin(-syaw) * ix + math.cos(-syaw) * iy + sy for (ix, iy) in zip(x_list, y_list)]
        yaw_list_ = [Geometry.regularize_orient(iyaw + syaw) for iyaw in yaw_list]

        return best_cost / self.max_curv, best_path.ctypes, x_list_, y_list_, yaw_list_

    def _cal_tau_omega(self, u: float, v: float, xi: float, eta: float, phi: float) -> Tuple[float, float]:
        """
        Helper to compute (tau, omega) for LRLR-family patterns.

        Args:
            u, v: Intermediate angular values from the base pattern.
            xi, eta: Coordinates derived from the normalized goal pose.
            phi: Normalized goal heading.

        Returns:
            tau, omega: Angular values used to complete the pattern.
        """
        delta = Geometry.regularize_orient(u - v)
        A = math.sin(u) - math.sin(delta)
        B = math.cos(u) - math.cos(delta) - 1.0

        t1 = math.atan2(eta * A - xi * B, xi * A + eta * B)
        t2 = 2.0 * (math.cos(delta) - math.cos(v) - math.cos(u)) + 3.0

        tau = Geometry.regularize_orient(t1 + math.pi) if t2 < 0 else Geometry.regularize_orient(t1)
        omega = Geometry.regularize_orient(tau - u + v - phi)
        return tau, omega