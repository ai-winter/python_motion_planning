"""
@file: bezier_curve.py
@author: Yang Haodong, Wu Maojia
@update: 2026.4.12
"""
from typing import List, Tuple, Dict, Any

import numpy as np
from scipy.special import comb

from python_motion_planning.traj_optimizer.base_curve_generator import BaseCurveGenerator


class Bezier(BaseCurveGenerator):
    """
    Class for Bezier curve generator.

    Args:
        *args: see the parent class.
        offset: The offset of control points (larger value yields sharper curvature).
        *kwargs: see the parent class.

    References:
        [1] https://en.wikipedia.org/wiki/B%C3%A9zier_curve

    Examples:
        >>> import math
        >>> generator = Bezier(step=0.1, offset=3.0)
        >>> points = [(0.0, 0.0, 0.0), (10.0, 10.0, -math.pi/2), (20.0, 5.0, math.pi/3)]
        >>> path, curve_info = generator.generate(points)
        >>> print(curve_info['success'])
        True
    """
    def __init__(self, *args,
                 offset: float = 3.0,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.offset = offset

    def __str__(self) -> str:
        return "Bezier Curve"

    def generate(self, points: List[Tuple[float, float, float]]) -> Tuple[List[Tuple[float, float]], Dict[str, Any]]:
        """
        Generate a concatenated Bezier curve through a list of poses.

        Args:
            points: A list of poses (x, y, yaw) in world frame.

        Returns:
            path: A list of (x, y) waypoints of the generated curve in world frame.
            curve_info: A dictionary containing the curve information (success, length).
        """
        if len(points) < 2:
            return [], {"success": False, "length": 0.0}

        path: List[Tuple[float, float]] = []
        for i in range(len(points) - 1):
            segment, _ = self._generate_segment(points[i], points[i + 1])
            path.extend([(float(pt[0]), float(pt[1])) for pt in segment])

        return path, {"success": True, "length": self.length(path)}

    def _generate_segment(self, start_pose: Tuple[float, float, float],
                          goal_pose: Tuple[float, float, float]
                          ) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
        """
        Generate a single Bezier curve segment between two poses.

        Args:
            start_pose: Initial pose (x, y, yaw).
            goal_pose: Target pose (x, y, yaw).

        Returns:
            segment: A list of points sampled from the Bezier curve.
            control_points: The control points of the Bezier curve.
        """
        sx, sy, _ = start_pose
        gx, gy, _ = goal_pose
        n_points = max(int(np.hypot(sx - gx, sy - gy) / self.step), 2)
        control_points = self._get_control_points(start_pose, goal_pose)

        segment = [self._bezier(t, control_points) for t in np.linspace(0, 1, n_points)]
        return segment, control_points

    def _bezier(self, t: float, control_points: List[Tuple[float, float]]) -> np.ndarray:
        """
        Calculate the Bezier curve point.

        Args:
            t: Scale factor in [0, 1].
            control_points: Control points of the Bezier curve.

        Returns:
            point: Point on the Bezier curve at the given t.
        """
        n = len(control_points) - 1
        control_points = np.array(control_points)
        return np.sum([comb(n, i) * t ** i * (1 - t) ** (n - i) * control_points[i]
                       for i in range(n + 1)], axis=0)

    def _get_control_points(self, start_pose: Tuple[float, float, float],
                            goal_pose: Tuple[float, float, float]) -> List[Tuple[float, float]]:
        """
        Calculate the control points heuristically from start and goal poses.

        Args:
            start_pose: Initial pose (x, y, yaw).
            goal_pose: Target pose (x, y, yaw).

        Returns:
            control_points: Control points of the Bezier curve.
        """
        sx, sy, syaw = start_pose
        gx, gy, gyaw = goal_pose

        dist = np.hypot(sx - gx, sy - gy) / self.offset
        return [(sx, sy),
                (sx + dist * np.cos(syaw), sy + dist * np.sin(syaw)),
                (gx - dist * np.cos(gyaw), gy - dist * np.sin(gyaw)),
                (gx, gy)]
