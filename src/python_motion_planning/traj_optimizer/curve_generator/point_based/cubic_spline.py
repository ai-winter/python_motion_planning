"""
@file: cubic_spline.py
@author: Yang Haodong, Wu Maojia
@update: 2026.4.12
"""
from typing import List, Tuple, Dict, Any
import math
import bisect

import numpy as np

from python_motion_planning.traj_optimizer.base_curve_generator import BaseCurveGenerator


class CubicSpline(BaseCurveGenerator):
    """
    Class for cubic spline curve generator.

    Args:
        *args: see the parent class.
        *kwargs: see the parent class.

    References:
        [1] https://en.wikipedia.org/wiki/Spline_(mathematics)#Algorithm_for_computing_natural_cubic_splines

    Examples:
        >>> generator = CubicSpline(step=0.1)
        >>> points = [(0.0, 0.0), (10.0, 10.0), (20.0, 5.0)]
        >>> path, curve_info = generator.generate(points)
        >>> print(curve_info['success'])
        True
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __str__(self) -> str:
        return "Cubic Spline"

    def generate(self, points: List[Tuple[float, ...]]) -> Tuple[List[Tuple[float, float]], Dict[str, Any]]:
        """
        Generate a cubic spline curve through a list of 2D points.

        Args:
            points: A list of 2D points (x, y) in world frame. If the points contain
                additional entries (e.g. yaw), only the first two values are used.

        Returns:
            path: A list of (x, y) waypoints of the generated curve in world frame.
            curve_info: A dictionary containing the curve information (success, length).
        """
        if len(points) < 2:
            return [], {"success": False, "length": 0.0}

        x_list = [float(p[0]) for p in points]
        y_list = [float(p[1]) for p in points]

        dx, dy = np.diff(x_list), np.diff(y_list)
        ds = [math.hypot(idx, idy) for (idx, idy) in zip(dx, dy)]
        s = [0.0]
        s.extend(np.cumsum(ds))
        t = np.arange(0, s[-1], self.step)

        path_x, _ = self._spline(s, x_list, t)
        path_y, _ = self._spline(s, y_list, t)

        path = [(float(ix), float(iy)) for ix, iy in zip(path_x, path_y)]
        return path, {"success": True, "length": self.length(path)}

    def _spline(self, x_list: List[float], y_list: List[float],
                t: np.ndarray) -> Tuple[List[float], List[float]]:
        """
        Build and evaluate a 1D natural cubic spline y = f(x).

        Args:
            x_list: Monotonically increasing x-coordinates of the control points.
            y_list: y-coordinates of the control points.
            t: Values of x at which to evaluate the spline.

        Returns:
            p: Values of the spline evaluated at t.
            dp: Values of the spline derivative evaluated at t.
        """
        a, b, c, d = y_list, [], [], []
        h = np.diff(x_list)
        num = len(x_list)

        A = np.zeros((num, num))
        for i in range(1, num - 1):
            A[i, i - 1] = h[i - 1]
            A[i, i] = 2.0 * (h[i - 1] + h[i])
            A[i, i + 1] = h[i]
        A[0, 0] = 1.0
        A[num - 1, num - 1] = 1.0

        B = np.zeros(num)
        for i in range(1, num - 1):
            B[i] = (3.0 * (a[i + 1] - a[i]) / h[i]
                    - 3.0 * (a[i] - a[i - 1]) / h[i - 1])

        c = np.linalg.solve(A, B)
        for i in range(num - 1):
            d.append((c[i + 1] - c[i]) / (3.0 * h[i]))
            b.append((a[i + 1] - a[i]) / h[i] - h[i] * (c[i + 1] + 2.0 * c[i]) / 3.0)

        p, dp = [], []
        for it in t:
            if it < x_list[0] or it > x_list[-1]:
                continue
            i = bisect.bisect(x_list, it) - 1
            i = min(max(i, 0), num - 2)
            dx = it - x_list[i]
            p.append(a[i] + b[i] * dx + c[i] * dx ** 2 + d[i] * dx ** 3)
            dp.append(b[i] + 2.0 * c[i] * dx + 3.0 * d[i] * dx ** 2)

        return p, dp
