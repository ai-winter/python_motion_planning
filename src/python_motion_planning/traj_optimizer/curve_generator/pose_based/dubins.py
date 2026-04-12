"""
@file: dubins_curve.py
@author: Yang Haodong, Wu Maojia
@update: 2026.4.12
"""
from typing import List, Tuple, Dict, Any
import math

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from python_motion_planning.traj_optimizer.base_curve_generator import BaseCurveGenerator
from python_motion_planning.common.utils.geometry import Geometry


class Dubins(BaseCurveGenerator):
    """
    Class for Dubins curve generator.

    Args:
        *args: see the parent class.
        max_curv: The maximum curvature of the curve.
        *kwargs: see the parent class.

    References:
        [1] On curves of minimal length with a constraint on average curvature, and with prescribed initial and terminal positions and tangents

    Examples:
        >>> import math
        >>> generator = Dubins(step=0.1, max_curv=1.0)
        >>> points = [(0.0, 0.0, 0.0), (10.0, 10.0, -math.pi/2), (20.0, 5.0, math.pi/3)]
        >>> path, curve_info = generator.generate(points)
        >>> print(curve_info['success'])
        True
    """
    def __init__(self, *args, max_curv: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_curv = max_curv

    def __str__(self) -> str:
        return "Dubins Curve"

    def generate(self, points: List[Tuple[float, float, float]]) -> Tuple[List[Tuple[float, float, float]], Dict[str, Any]]:
        """
        Generate a concatenated Dubins curve through a list of poses.

        Args:
            points: A list of poses (x, y, yaw) in world frame.

        Returns:
            path: A list of (x, y, yaw) waypoints of the generated curve in world frame.
            curve_info: A dictionary containing the curve information (success, length).
        """
        if len(points) < 2:
            return [], {"success": False, "length": 0.0}

        path: List[Tuple[float, float, float]] = []
        total_cost = 0.0
        for i in range(len(points) - 1):
            best_cost, _, x_list, y_list, yaw_list = self._generate_segment(
                points[i], points[i + 1])
            if best_cost is None:
                return [], {"success": False, "length": 0.0}
            total_cost += best_cost / self.max_curv

            start = 1 if i > 0 else 0
            for x, y, yaw in zip(x_list[start:], y_list[start:], yaw_list[start:]):
                path.append((float(x), float(y), float(yaw)))

        total_cost = float(total_cost)

        return path, {"success": True, "length": total_cost}

    def _lsl(self, alpha: float, beta: float, dist: float):
        """
        Left-Straight-Left generation mode.

        Args:
            alpha: Initial heading of pose (0, 0, alpha).
            beta: Goal heading of pose (dist, 0, beta).
            dist: The distance between the initial and goal poses.

        Returns:
            t, p, q: Moving length of segments.
            mode: Motion mode.
        """
        sin_a, sin_b, cos_a, cos_b, _, cos_a_b = self.trigonometric(alpha, beta)

        p_lsl = 2 + dist ** 2 - 2 * cos_a_b + 2 * dist * (sin_a - sin_b)
        if p_lsl < 0:
            return None, None, None, ["L", "S", "L"]
        p_lsl = math.sqrt(p_lsl)

        t_lsl = Geometry.mod_to_2pi(-alpha + math.atan2(cos_b - cos_a, dist + sin_a - sin_b))
        q_lsl = Geometry.mod_to_2pi(beta - math.atan2(cos_b - cos_a, dist + sin_a - sin_b))
        return t_lsl, p_lsl, q_lsl, ["L", "S", "L"]

    def _rsr(self, alpha: float, beta: float, dist: float):
        """
        Right-Straight-Right generation mode.

        Args:
            alpha: Initial heading of pose (0, 0, alpha).
            beta: Goal heading of pose (dist, 0, beta).
            dist: The distance between the initial and goal poses.

        Returns:
            t, p, q: Moving length of segments.
            mode: Motion mode.
        """
        sin_a, sin_b, cos_a, cos_b, _, cos_a_b = self.trigonometric(alpha, beta)

        p_rsr = 2 + dist ** 2 - 2 * cos_a_b + 2 * dist * (sin_b - sin_a)
        if p_rsr < 0:
            return None, None, None, ["R", "S", "R"]
        p_rsr = math.sqrt(p_rsr)

        t_rsr = Geometry.mod_to_2pi(alpha - math.atan2(cos_a - cos_b, dist - sin_a + sin_b))
        q_rsr = Geometry.mod_to_2pi(-beta + math.atan2(cos_a - cos_b, dist - sin_a + sin_b))
        return t_rsr, p_rsr, q_rsr, ["R", "S", "R"]

    def _lsr(self, alpha: float, beta: float, dist: float):
        """
        Left-Straight-Right generation mode.

        Args:
            alpha: Initial heading of pose (0, 0, alpha).
            beta: Goal heading of pose (dist, 0, beta).
            dist: The distance between the initial and goal poses.

        Returns:
            t, p, q: Moving length of segments.
            mode: Motion mode.
        """
        sin_a, sin_b, cos_a, cos_b, _, cos_a_b = self.trigonometric(alpha, beta)

        p_lsr = -2 + dist ** 2 + 2 * cos_a_b + 2 * dist * (sin_a + sin_b)
        if p_lsr < 0:
            return None, None, None, ["L", "S", "R"]
        p_lsr = math.sqrt(p_lsr)

        t_lsr = Geometry.mod_to_2pi(-alpha + math.atan2(-cos_a - cos_b, dist + sin_a + sin_b) - math.atan2(-2.0, p_lsr))
        q_lsr = Geometry.mod_to_2pi(-beta + math.atan2(-cos_a - cos_b, dist + sin_a + sin_b) - math.atan2(-2.0, p_lsr))
        return t_lsr, p_lsr, q_lsr, ["L", "S", "R"]

    def _rsl(self, alpha: float, beta: float, dist: float):
        """
        Right-Straight-Left generation mode.

        Args:
            alpha: Initial heading of pose (0, 0, alpha).
            beta: Goal heading of pose (dist, 0, beta).
            dist: The distance between the initial and goal poses.

        Returns:
            t, p, q: Moving length of segments.
            mode: Motion mode.
        """
        sin_a, sin_b, cos_a, cos_b, _, cos_a_b = self.trigonometric(alpha, beta)

        p_rsl = -2 + dist ** 2 + 2 * cos_a_b - 2 * dist * (sin_a + sin_b)
        if p_rsl < 0:
            return None, None, None, ["R", "S", "L"]
        p_rsl = math.sqrt(p_rsl)

        t_rsl = Geometry.mod_to_2pi(alpha - math.atan2(cos_a + cos_b, dist - sin_a - sin_b) + math.atan2(2.0, p_rsl))
        q_rsl = Geometry.mod_to_2pi(beta - math.atan2(cos_a + cos_b, dist - sin_a - sin_b) + math.atan2(2.0, p_rsl))
        return t_rsl, p_rsl, q_rsl, ["R", "S", "L"]

    def _rlr(self, alpha: float, beta: float, dist: float):
        """
        Right-Left-Right generation mode.

        Args:
            alpha: Initial heading of pose (0, 0, alpha).
            beta: Goal heading of pose (dist, 0, beta).
            dist: The distance between the initial and goal poses.

        Returns:
            t, p, q: Moving length of segments.
            mode: Motion mode.
        """
        sin_a, sin_b, cos_a, cos_b, _, cos_a_b = self.trigonometric(alpha, beta)

        p_rlr = (6.0 - dist ** 2 + 2.0 * cos_a_b + 2.0 * dist * (sin_a - sin_b)) / 8.0
        if abs(p_rlr) > 1.0:
            return None, None, None, ["R", "L", "R"]
        p_rlr = Geometry.mod_to_2pi(2 * math.pi - math.acos(p_rlr))

        t_rlr = Geometry.mod_to_2pi(alpha - math.atan2(cos_a - cos_b, dist - sin_a + sin_b) + p_rlr / 2.0)
        q_rlr = Geometry.mod_to_2pi(alpha - beta - t_rlr + p_rlr)
        return t_rlr, p_rlr, q_rlr, ["R", "L", "R"]

    def _lrl(self, alpha: float, beta: float, dist: float):
        """
        Left-Right-Left generation mode.

        Args:
            alpha: Initial heading of pose (0, 0, alpha).
            beta: Goal heading of pose (dist, 0, beta).
            dist: The distance between the initial and goal poses.

        Returns:
            t, p, q: Moving length of segments.
            mode: Motion mode.
        """
        sin_a, sin_b, cos_a, cos_b, _, cos_a_b = self.trigonometric(alpha, beta)

        p_lrl = (6.0 - dist ** 2 + 2.0 * cos_a_b + 2.0 * dist * (sin_b - sin_a)) / 8.0
        if abs(p_lrl) > 1.0:
            return None, None, None, ["L", "R", "L"]
        p_lrl = Geometry.mod_to_2pi(2 * math.pi - math.acos(p_lrl)  )

        t_lrl = Geometry.mod_to_2pi(-alpha + math.atan2(-cos_a + cos_b, dist + sin_a - sin_b) + p_lrl / 2.0)
        q_lrl = Geometry.mod_to_2pi(beta - alpha - t_lrl + p_lrl)
        return t_lrl, p_lrl, q_lrl, ["L", "R", "L"]

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
        Generate a single Dubins curve segment between two poses.

        Args:
            start_pose: Initial pose (x, y, yaw).
            goal_pose: Target pose (x, y, yaw).

        Returns:
            best_cost: Best planning path length.
            best_mode: Best motion modes.
            x_list: Trajectory of x.
            y_list: Trajectory of y.
            yaw_list: Trajectory of yaw.
        """
        sx, sy, syaw = start_pose
        gx, gy, gyaw = goal_pose

        dx = gx - sx
        dy = gy - sy
        
        cos_s = math.cos(syaw)
        sin_s = math.sin(syaw)
        local_gx = dx * cos_s + dy * sin_s
        local_gy = -dx * sin_s + dy * cos_s
        local_gyaw = gyaw - syaw
        
        dist = math.hypot(local_gx, local_gy) * self.max_curv
        theta = Geometry.mod_to_2pi(math.atan2(local_gy, local_gx))
        
        alpha = Geometry.mod_to_2pi(-theta)
        beta = Geometry.mod_to_2pi(local_gyaw - theta)

        planners = [self._lsl, self._rsr, self._lsr, self._rsl, self._rlr, self._lrl]
        best_t, best_p, best_q, best_mode, best_cost = None, None, None, None, float("inf")

        for planner in planners:
            t, p, q, mode = planner(alpha, beta, dist)
            if t is None:
                continue
            cost = abs(t) + abs(p) + abs(q)
            if best_cost > cost:
                best_t, best_p, best_q, best_mode, best_cost = t, p, q, mode, cost

        if best_mode is None:
            return None, None, [], [], []

        segments = [best_t, best_p, best_q]
        points_num = int(sum(segments) / self.step) + len(segments) + 3
        x_list = [0.0 for _ in range(points_num)]
        y_list = [0.0 for _ in range(points_num)]
        yaw_list = [0.0 for _ in range(points_num)]

        idx = 0
        for mode_, seg_length in zip(best_mode, segments):
            d_length = self.step if seg_length > 0.0 else -self.step
            current_x, current_y, current_yaw = x_list[idx], y_list[idx], yaw_list[idx]
            length = d_length
            while abs(length) <= abs(seg_length):
                idx += 1
                current_x, current_y, current_yaw = self._interpolate(
                    mode_, d_length, (current_x, current_y, current_yaw)
                )
                x_list[idx], y_list[idx], yaw_list[idx] = current_x, current_y, current_yaw
                length += d_length
            
            idx += 1
            remainder = seg_length - (length - d_length)
            x_list[idx], y_list[idx], yaw_list[idx] = self._interpolate(
                mode_, remainder, (x_list[idx-1], y_list[idx-1], y_list[idx-1])
            )

        x_list = x_list[:idx + 1]
        y_list = y_list[:idx + 1]
        yaw_list = yaw_list[:idx + 1]

        if len(x_list) <= 1:
            return None, None, [], [], []

        rot = Rot.from_euler('z', syaw).as_matrix()[0:2, 0:2]
        converted_xy = rot @ np.stack([x_list, y_list])
        x_list = (converted_xy[0, :] + sx).tolist()
        y_list = (converted_xy[1, :] + sy).tolist()
        yaw_list = [Geometry.regularize_orient(i_yaw + syaw) for i_yaw in yaw_list]

        return best_cost, best_mode, x_list, y_list, yaw_list
    
    def trigonometric(self, alpha: float, beta: float) -> Tuple[float, float, float, float, float, float]:
        """
        Calculate trigonometric values for alpha and beta.
        
        Args:
            alpha: Initial heading angle.
            beta: Goal heading angle.
            
        Returns:
            sin_a, sin_b, cos_a, cos_b, cos_ab, cos_a_b: Trigonometric values.
        """
        sin_a = math.sin(alpha)
        sin_b = math.sin(beta)
        cos_a = math.cos(alpha)
        cos_b = math.cos(beta)
        cos_ab = math.cos(alpha - beta)
        cos_a_b = math.cos(alpha + beta) if hasattr(self, '_use_sum') else cos_ab
        return sin_a, sin_b, cos_a, cos_b, cos_ab, cos_ab