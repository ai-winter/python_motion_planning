"""
@file: bspline_curve.py
@author: Yang Haodong, Wu Maojia
@update: 2026.4.12
"""
from typing import List, Tuple, Dict, Any
import math

import numpy as np

from python_motion_planning.traj_optimizer.base_curve_generator import BaseCurveGenerator


class BSpline(BaseCurveGenerator):
    """
    Class for B-Spline curve generator.

    Args:
        *args: see the parent class.
        k: Degree of the B-Spline curve.
        param_mode: Parameter selection mode, one of {"centripetal", "chord_length", "uniform_spaced"}.
        spline_mode: B-Spline construction mode, one of {"interpolation", "approximation"}.
        *kwargs: see the parent class.

    References:
        [1] https://en.wikipedia.org/wiki/B-spline

    Examples:
        >>> generator = BSpline(step=0.1, k=3)
        >>> points = [(0.0, 0.0), (10.0, 10.0), (20.0, 5.0)]
        >>> path, curve_info = generator.generate(points)
        >>> print(curve_info['success'])
        True
    """
    def __init__(self, *args,
                 k: int = 3,
                 param_mode: str = "centripetal",
                 spline_mode: str = "interpolation",
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.k = k

        if param_mode not in ("centripetal", "chord_length", "uniform_spaced"):
            raise ValueError("Parameter selection mode error!")
        self.param_mode = param_mode

        if spline_mode not in ("interpolation", "approximation"):
            raise ValueError("Spline mode selection error!")
        self.spline_mode = spline_mode

    def __str__(self) -> str:
        return "B-Spline Curve"

    def generate(self, points: List[Tuple[float, ...]]) -> Tuple[List[Tuple[float, float]], Dict[str, Any]]:
        """
        Generate a B-Spline curve through a list of 2D points.

        Args:
            points: A list of 2D points (x, y) in world frame. If the points contain
                additional entries (e.g. yaw), only the first two values are used.

        Returns:
            path: A list of (x, y) waypoints of the generated curve in world frame.
            curve_info: A dictionary containing the curve information (success, length).
        """
        if len(points) < 2:
            return [], {"success": False, "length": 0.0}

        points_2d = [(float(p[0]), float(p[1])) for p in points]

        t = np.linspace(0, 1, int(1 / self.step))
        params = self._param_selection(points_2d)
        knot = self._knot_generation(params, len(points_2d))

        if self.spline_mode == "interpolation":
            control_pts = self._interpolation(points_2d, params, knot)
        else:
            control_pts = self._approximation(points_2d, params, knot)
            h = len(control_pts)
            new_points = [(control_pts[i][0], control_pts[i][1]) for i in range(h)]
            params = self._param_selection(new_points)
            knot = self._knot_generation(params, h)

        curve = self._generate_curve(t, self.k, knot, control_pts)
        path = [(float(pt[0]), float(pt[1])) for pt in curve]

        return path, {"success": True, "length": self.length(path)}

    def _base_function(self, i: int, k: int, t: float, knot: List[float]) -> float:
        """
        Calculate the base function using the Cox-de Boor recursion.

        Args:
            i: The index of the base function.
            k: The degree of the curve.
            t: Parameter value.
            knot: The knot vector.

        Returns:
            Nik_t: The value of the base function Nik(t).
        """
        Nik_t = 0.0
        if k == 0:
            Nik_t = 1.0 if knot[i] <= t < knot[i + 1] else 0.0
        else:
            length1 = knot[i + k] - knot[i]
            length2 = knot[i + k + 1] - knot[i + 1]
            if not length1 and not length2:
                Nik_t = 0.0
            elif not length1:
                Nik_t = (knot[i + k + 1] - t) / length2 * self._base_function(i + 1, k - 1, t, knot)
            elif not length2:
                Nik_t = (t - knot[i]) / length1 * self._base_function(i, k - 1, t, knot)
            else:
                Nik_t = ((t - knot[i]) / length1 * self._base_function(i, k - 1, t, knot)
                         + (knot[i + k + 1] - t) / length2 * self._base_function(i + 1, k - 1, t, knot))
        return Nik_t

    def _param_selection(self, points: List[Tuple[float, float]]) -> List[float]:
        """
        Calculate the parameters of the given points using the selected method.

        Args:
            points: A list of 2D points.

        Returns:
            parameters: The parameters of the given points.
        """
        n = len(points)
        x_list = [pt[0] for pt in points]
        y_list = [pt[1] for pt in points]
        dx, dy = np.diff(x_list), np.diff(y_list)

        if self.param_mode == "uniform_spaced":
            return np.linspace(0, 1, n).tolist()

        if self.param_mode == "chord_length":
            parameters = np.zeros(n)
            s = np.cumsum([math.hypot(idx, idy) for (idx, idy) in zip(dx, dy)])
            for i in range(1, n):
                parameters[i] = s[i - 1] / s[-1]
            return parameters.tolist()

        # centripetal
        alpha = 0.5
        s = np.cumsum([math.pow(math.hypot(idx, idy), alpha) for (idx, idy) in zip(dx, dy)])
        parameters = np.zeros(n)
        for i in range(1, n):
            parameters[i] = s[i - 1] / s[-1]
        return parameters.tolist()

    def _knot_generation(self, param: List[float], n: int) -> List[float]:
        """
        Generate the knot vector.

        Args:
            param: The parameters of the given points.
            n: The number of data points.

        Returns:
            knot: The knot vector.
        """
        m = n + self.k + 1
        knot = np.zeros(m)
        for i in range(n, m):
            knot[i] = 1
        for i in range(self.k + 1, n):
            for j in range(i - self.k, i):
                knot[i] = knot[i] + param[j]
            knot[i] = knot[i] / self.k
        return knot.tolist()

    def _interpolation(self, points: List[Tuple[float, float]], param: List[float],
                       knot: List[float]) -> np.ndarray:
        """
        Build a B-Spline curve of degree k defined by N control points that passes
        through all N data points.

        Args:
            points: A list of 2D points.
            param: The parameters of the given points.
            knot: The knot vector.

        Returns:
            control_points: The control points of the resulting B-Spline.
        """
        n = len(points)
        N = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                N[i][j] = self._base_function(j, self.k, param[i], knot)
        N[n - 1][n - 1] = 1
        N_inv = np.linalg.inv(N)

        D = np.array(points)
        return N_inv @ D

    def _approximation(self, points: List[Tuple[float, float]], param: List[float],
                       knot: List[float]) -> np.ndarray:
        """
        Build a B-Spline curve of degree k defined by H control points (H < N) that
        passes through the first and last data points and approximates the rest in
        the least-square sense.

        Args:
            points: A list of 2D points.
            param: The parameters of the given points.
            knot: The knot vector.

        Returns:
            control_points: The control points of the resulting B-Spline.
        """
        n = len(points)
        D = np.array(points)

        h = n - 1

        N = np.zeros((n, h))
        for i in range(n):
            for j in range(h):
                N[i][j] = self._base_function(j, self.k, param[i], knot)
        N_ = N[1: n - 1, 1: h - 1]

        qk = np.zeros((n - 2, 2))
        for i in range(1, n - 1):
            qk[i - 1] = D[i, :] - N[i][0] * D[0, :] - N[i][h - 1] * D[-1, :]
        Q = N_.T @ qk

        P = np.linalg.inv(N_.T @ N_) @ Q
        P = np.insert(P, 0, D[0, :], axis=0)
        P = np.insert(P, len(P), D[-1, :], axis=0)
        return P

    def _generate_curve(self, t: np.ndarray, k: int, knot: List[float],
                        control_pts: np.ndarray) -> np.ndarray:
        """
        Sample points on the B-Spline curve.

        Args:
            t: The parameter values to sample at.
            k: The degree of the B-Spline curve.
            knot: The knot vector.
            control_pts: The control points.

        Returns:
            curve: Sampled points along the B-Spline curve.
        """
        N = np.zeros((len(t), len(control_pts)))

        for i in range(len(t)):
            for j in range(len(control_pts)):
                N[i][j] = self._base_function(j, k, t[i], knot)
        N[len(t) - 1][len(control_pts) - 1] = 1

        return N @ control_pts
