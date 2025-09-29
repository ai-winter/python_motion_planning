from typing import List, Tuple
import math

import numpy as np

from python_motion_planning.common.utils.geometry import Geometry
from python_motion_planning.common.utils.frame_transformer import FrameTransformer
from .base_controller import BaseController

class PathTracker(BaseController):
    """
    Class of path-tracking controller.

    Args:
        *args: see the parent class.
        lookahead_distance: lookahead distance for path tracking
        k_theta: weight of theta error
        pose_interp: whether to interpolate between poses. if not, poses on the segments are last pose
        **kwargs: see the parent class.
    """
    def __init__(self,
                 *args,
                 lookahead_distance: float = 1.5,
                 k_theta: float = 0.8,
                 pose_interp: bool = False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.lookahead_distance = lookahead_distance
        self.k_theta = k_theta
        self.pose_interp = pose_interp
        self.current_target_index = 0

    def reset(self):
        """
        Reset the controller to initial state.
        """
        super().reset()
        self.current_target_index = 0

    def get_action(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get action from observation.

        Args:
            obs: observation in world frame ([pos, orient, lin_vel, ang_vel])

        Returns:
            action: action in robot frame ([lin_acc, ang_acc])
            target_pose: lookahead pose ([pos, orient])
        """
        if self.goal is None:
            return np.zeros(self.action_space.shape), self.goal

        pose, vel, pos, orient, lin_vel, ang_vel = self.get_pose_velocity(obs)

        # Find the lookahead pose
        target_pose = self._get_lookahead_pose(pos)

        desired_vel = self._get_desired_vel(target_pose, pose)
        desired_vel = self._stop_if_reached(desired_vel, pose)
        robot_vel = FrameTransformer.vel_world_to_robot(self.dim, vel, orient)
        action = self._get_desired_action(desired_vel, robot_vel, orient)

        return action, target_pose

    def _get_desired_vel(self, target_pose: np.ndarray, cur_pose: np.ndarray) -> np.ndarray:
        """
        Calculate the desired velocity in robot frame.

        Args:
            target_pose: target pose in world frame
            cur_pose: current pose in world frame

        Returns:
            desired_vel: desired velocity in robot frame
        """
        rel_pose = FrameTransformer.pose_world_to_robot(self.dim, target_pose, cur_pose)

        lin_direction = rel_pose[:self.dim]
        angle_lin_diff = np.array([math.atan2(lin_direction[1], lin_direction[0])])
        angle_ang_diff = rel_pose[self.dim:]
        ang_direction = Geometry.regularize_orient(
            self.k_theta * angle_lin_diff +
            (1.0 - self.k_theta) * angle_ang_diff
            )

        lin_distance = np.linalg.norm(lin_direction)
        ang_distance = np.linalg.norm(ang_direction)
        if lin_distance > 1e-6:
            lin_direction /= lin_distance
        if ang_distance > 1e-6:
            ang_direction /= ang_distance

        max_lin_speed = np.linalg.norm(self.action_space.high[:self.dim])
        max_ang_speed = np.linalg.norm(self.action_space.high[self.dim:])

        desired_lin_speed = min(lin_distance / self.dt, max_lin_speed)
        desired_ang_speed = min(ang_distance / self.dt, max_ang_speed)

        desired_lin_vel = lin_direction * desired_lin_speed
        desired_ang_vel = ang_direction * desired_ang_speed

        desired_vel = np.concatenate([desired_lin_vel, desired_ang_vel])
        desired_vel = self.clip_velocity(desired_vel)

        return desired_vel

    def _get_desired_action(self, desired_vel: np.ndarray, vel: np.ndarray, orient: np.ndarray) -> np.ndarray:
        """
        Calculates the action to be taken to reach the desired velocity.

        Args:
            desired_vel: Desired velocity in robot frame.
            vel: Current velocity in robot frame.
            orient: Current orientation in world frame.

        Returns:
            np.ndarray: Action to be taken to reach the desired velocity.
        """
        action = (desired_vel - vel) / self.dt
        action = self.clip_action(action)
        return action

    def _get_lookahead_pose(self, pos: np.ndarray) -> np.ndarray:
        """
        Get the lookahead pose (x, y, theta) on the path.
        Find the intersection point of the path with a circle centered at the robot's position with radius lookahead_distance.
        If the goal pose is within the lookahead distance, return the goal pose.
        If there are multiple intersections, return the one ahead of the robot along the path.
        If there are no intersections, return the closest pose on the path.

        Args:
            pos: robot position (x, y)

        Returns:
            lookahead_pose: (x, y, theta)
        """
        path = np.array(self.path)  # shape (N, 3)
        xy_path = path[:, :self.dim]
        r = self.lookahead_distance
        end_pose = path[-1]

        # check goal distance
        if np.dot(end_pose[:self.dim] - pos, end_pose[:self.dim] - pos) <= r * r + 1e-6:
            return end_pose

        # collect intersections
        candidates = []
        for i in range(len(xy_path) - 1):
            p1, p2 = xy_path[i], xy_path[i + 1]
            theta1, theta2 = path[i, 2], path[i + 1, 2]
            candidates.extend(
                (i + t, pose)
                for t, pose in self._circle_segment_intersections(pos, r, p1, p2, theta1, theta2)
            )

        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[-1][1]

        # fallback: closest point on path
        return self._closest_point_on_path(pos, path)

    def _circle_segment_intersections(self, pos: np.ndarray, r: float, p1: np.ndarray, p2: np.ndarray, theta1: float, theta2: float):
        """
        Find intersections between circle (center pos, radius r)
        and line segment p1-p2 (with orientation interpolation).
        
        Args:
            pos: np.ndarray, circle center
            r: float, circle radius
            p1: np.ndarray, line segment start
            p2: np.ndarray, line segment end
            theta1: float, line segment start orientation
            theta2: float, line segment end orientation

        Returns:
            List of (t, intersect_pose) where t is segment ratio in [0,1]
        """
        d = p2 - p1
        v = pos - p1
        a = np.dot(d, d)
        if a < 1e-10:
            return []

        b = -2 * np.dot(v, d)
        c = np.dot(v, v) - r * r
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return []

        sqrt_d = np.sqrt(discriminant)
        t1 = (-b + sqrt_d) / (2 * a)
        t2 = (-b - sqrt_d) / (2 * a)

        intersections = []
        for t in (t1, t2):
            if 0.0 <= t <= 1.0:
                xy = p1 + t * d
                theta = theta1
                if self.pose_interp:
                    theta += t * (theta2 - theta1)
                theta = Geometry.regularize_orient(theta)
                pose = np.array([xy[0], xy[1], theta])
                intersections.append((t, pose))

        return intersections

    def _closest_point_on_path(self, pos: np.ndarray, path: np.ndarray) -> np.ndarray:
        """
        Find the closest point (with theta interpolation) on a polyline path to pos.
        
        Args:
            pos: query point (x, y)
            path: array shape (N, 3), columns = (x, y, theta)

        Returns:
            closest_pose: np.ndarray (x, y, theta)
        """
        xy_path = path[:, :2]
        min_dist_sq = float("inf")
        closest_pose = path[0]

        for i in range(len(xy_path) - 1):
            p1, p2 = xy_path[i], xy_path[i + 1]
            d = p2 - p1
            v = pos - p1
            a = np.dot(d, d)
            if a < 1e-10:
                continue

            t = np.dot(v, d) / a
            if t < 0.0:
                proj = p1
                theta_proj = path[i, 2]
            elif t > 1.0:
                proj = p2
                theta_proj = path[i + 1, 2]
            else:
                proj = p1 + y * d
                theta_proj = path[i, 2]
                if self.pose_interp:
                    theta_proj += t * (path[i + 1, 2] - path[i, 2])
                theta_proj = Geometry.regularize_orient(theta_proj)

            dist_sq = np.dot(pos - proj, pos - proj)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_pose = np.array([proj[0], proj[1], theta_proj])

        return closest_pose
