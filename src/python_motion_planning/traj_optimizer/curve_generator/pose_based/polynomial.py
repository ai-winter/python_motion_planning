"""
@file: polynomial_curve.py
@author: Yang Haodong, Wu Maojia
@update: 2026.4.12
"""
from typing import List, Tuple, Dict, Any
import math

import numpy as np

from python_motion_planning.traj_optimizer.base_curve_generator import BaseCurveGenerator


class Polynomial(BaseCurveGenerator):
    """
    Class for quintic polynomial curve generator.

    Args:
        *args: see the parent class.
        max_acc: Maximum allowed acceleration magnitude.
        max_jerk: Maximum allowed jerk magnitude.
        *args: see the parent class.

    References:
        [1] https://en.wikipedia.org/wiki/Polynomial_trajectory

    Examples:
        >>> import math
        >>> generator = Polynomial(step=0.1, max_acc=2.82, max_jerk=10.0)
        >>> points = [(0.0, 0.0, 0.0), (10.0, 10.0, -math.pi/2), (20.0, 5.0, math.pi/3)]
        >>> path, curve_info = generator.generate(points)
        >>> print(curve_info['success'])
        True
    """
    def __init__(self, *args, 
                 max_acc: float = 2.82, 
                 max_jerk: float = 10.0,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_acc = max_acc
        self.max_jerk = max_jerk
        self.dt = 0.1
        self.t_min = 1
        self.t_max = 30

    def __str__(self) -> str:
        return "Quintic Polynomial Curve"

    class _Poly:
        """
        Quintic polynomial solver for a single dimension.
        """
        def __init__(self, state0: tuple, state1: tuple, t: float) -> None:
            x0, v0, a0 = state0
            xt, vt, at = state1

            A = np.array([[t ** 3, t ** 4, t ** 5],
                          [3 * t ** 2, 4 * t ** 3, 5 * t ** 4],
                          [6 * t, 12 * t ** 2, 20 * t ** 3]])
            b = np.array([xt - x0 - v0 * t - a0 * t ** 2 / 2,
                          vt - v0 - a0 * t,
                          at - a0])
            X = np.linalg.solve(A, b)

            self.p0 = x0
            self.p1 = v0
            self.p2 = a0 / 2.0
            self.p3 = X[0]
            self.p4 = X[1]
            self.p5 = X[2]

        def x(self, t: float) -> float:
            return (self.p0 + self.p1 * t + self.p2 * t ** 2
                    + self.p3 * t ** 3 + self.p4 * t ** 4 + self.p5 * t ** 5)

        def dx(self, t: float) -> float:
            return (self.p1 + 2 * self.p2 * t + 3 * self.p3 * t ** 2
                    + 4 * self.p4 * t ** 3 + 5 * self.p5 * t ** 4)

        def ddx(self, t: float) -> float:
            return 2 * self.p2 + 6 * self.p3 * t + 12 * self.p4 * t ** 2 + 20 * self.p5 * t ** 3

        def dddx(self, t: float) -> float:
            return 6 * self.p3 + 24 * self.p4 * t + 60 * self.p5 * t ** 2

    class _Trajectory:
        """
        Container for a polynomial trajectory.
        """
        def __init__(self):
            self.clear()

        def clear(self):
            self.time = []
            self.x = []
            self.y = []
            self.yaw = []
            self.v = []
            self.a = []
            self.jerk = []

        @property
        def size(self) -> int:
            assert (len(self.time) == len(self.x) == len(self.y) == len(self.yaw)
                    == len(self.v) == len(self.a) == len(self.jerk)), \
                "Unequal dimensions of each attribute, this should not happen."
            return len(self.time)

    def generate(self, points: List[Tuple[float, ...]]) -> Tuple[List[Tuple[float, float, float]], Dict[str, Any]]:
        """
        Generate a concatenated quintic polynomial curve through a list of poses.

        Args:
            points: A list of poses (x, y, yaw) in world frame. Optionally the
                points may also carry velocity and acceleration, i.e.
                (x, y, yaw, v, a). Missing values are filled heuristically.

        Returns:
            path: A list of (x, y, yaw) waypoints of the generated curve in world frame.
            curve_info: A dictionary containing the curve information (success, length).
        """
        if len(points) < 2:
            return [], {"success": False, "length": 0.0}

        states: List[Tuple[float, float, float, float, float]] = []
        for i, pt in enumerate(points):
            if len(pt) >= 5:
                states.append((float(pt[0]), float(pt[1]), float(pt[2]), float(pt[3]), float(pt[4])))
            elif len(pt) == 3:
                v = 0.0 if (i == 0 or i == len(points) - 1) else 1.0
                states.append((float(pt[0]), float(pt[1]), float(pt[2]), v, 0.0))
            else:
                raise ValueError("Points must be (x, y, yaw) or (x, y, yaw, v, a).")

        path: List[Tuple[float, float, float]] = []
        for i in range(len(states) - 1):
            traj = self._generate_segment(states[i], states[i + 1])
            for j in range(traj.size):
                path.append((float(traj.x[j]), float(traj.y[j]), float(traj.yaw[j])))

        success = len(path) > 0
        return path, {"success": success, "length": self.length(path) if success else 0.0}

    def _generate_segment(self, start_state: Tuple[float, float, float, float, float],
                          goal_state: Tuple[float, float, float, float, float]) -> "_Trajectory":
        """
        Generate a single quintic polynomial segment between two states.

        Args:
            start_state: Initial state (x, y, yaw, v, a).
            goal_state: Target state (x, y, yaw, v, a).

        Returns:
            traj: The first trajectory that satisfies the acceleration and jerk constraints.
        """
        sx, sy, syaw, sv, sa = start_state
        gx, gy, gyaw, gv, ga = goal_state

        sv_x, sv_y = sv * math.cos(syaw), sv * math.sin(syaw)
        gv_x, gv_y = gv * math.cos(gyaw), gv * math.sin(gyaw)

        sa_x, sa_y = sa * math.cos(syaw), sa * math.sin(syaw)
        ga_x, ga_y = ga * math.cos(gyaw), ga * math.sin(gyaw)

        traj = self._Trajectory()

        for T in np.arange(self.t_min, self.t_max, self.step):
            x_psolver = self._Poly((sx, sv_x, sa_x), (gx, gv_x, ga_x), T)
            y_psolver = self._Poly((sy, sv_y, sa_y), (gy, gv_y, ga_y), T)

            for t in np.arange(0.0, T + self.dt, self.dt):
                traj.time.append(t)
                traj.x.append(x_psolver.x(t))
                traj.y.append(y_psolver.x(t))

                vx, vy = x_psolver.dx(t), y_psolver.dx(t)
                traj.v.append(math.hypot(vx, vy))
                traj.yaw.append(math.atan2(vy, vx))

                ax, ay = x_psolver.ddx(t), y_psolver.ddx(t)
                a = math.hypot(ax, ay)
                if len(traj.v) >= 2 and traj.v[-1] - traj.v[-2] < 0.0:
                    a *= -1
                traj.a.append(a)

                jx, jy = x_psolver.dddx(t), y_psolver.dddx(t)
                j = math.hypot(jx, jy)
                if len(traj.a) >= 2 and traj.a[-1] - traj.a[-2] < 0.0:
                    j *= -1
                traj.jerk.append(j)

            if (max(np.abs(traj.a)) <= self.max_acc
                    and max(np.abs(traj.jerk)) <= self.max_jerk):
                return traj
            traj.clear()

        return traj
