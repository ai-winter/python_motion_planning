from typing import List, Tuple
import math

import numpy as np

from python_motion_planning.common.utils.geometry import Geometry
from python_motion_planning.common.utils.frame_transformer import FrameTransformer
from .base_controller import BaseController

class PurePursuit(BaseController):
    """
    Class of Pure Pursuit path-tracking controller.

    Parameters:
        observation_space: observation space ([pos, orient, lin_vel, ang_vel])
        action_space: action space ([lin_acc, ang_acc])
        dt: time step for control
        path: path to follow
        max_speed: maximum speed of the robot
        lookahead_distance: lookahead distance for path tracking
        k_theta: weight of theta error
    """
    def __init__(self,
                 observation_space,
                 action_space,
                 dt: float,
                 path: List[Tuple[float, ...]] = [],
                 max_speed: float = np.inf,
                 lookahead_distance: float = 2.0,
                 k_theta: float = 0.7):
        super().__init__(observation_space, action_space, dt, path, max_speed)
        self.lookahead_distance = lookahead_distance
        self.k_theta = k_theta
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

        Parameters:
            obs: observation ([pos, orient, lin_vel, ang_vel])

        Returns:
            action: action ([lin_acc, ang_acc])
            target_pose: lookahead pose ([pos, orient])
        """
        if self.goal is None:
            return np.zeros(self.action_space.shape), self.goal

        pose, vel, pos, orient, lin_vel, ang_vel = self.get_pose_velocity(obs)

        # Find the lookahead pose
        target_pose = self._get_lookahead_pose(pos)

        # print(vel)
        print(target_pose, pose)
        desired_vel = self._get_desired_vel(target_pose, pose)
        # print(desired_vel)
        # print(desired_vel - vel)
        action = self._get_desired_action(desired_vel, vel, orient)
        # print(acc)
        # acc = FrameTransformer.vel_world_to_robot(self.dim, acc, orient)
        # print(acc, orient)
        print()

        return action, target_pose

    def _get_desired_vel(self, target_pose: np.ndarray, cur_pose: np.ndarray) -> np.ndarray:
        """
        Calculate the desired velocity in robot frame.
        """
        rel_pose = FrameTransformer.pose_world_to_robot(self.dim, target_pose, cur_pose)

        lin_direction = rel_pose[:self.dim]
        angle_lin_diff = np.array([math.atan2(lin_direction[1], lin_direction[0])])
        angle_ang_diff = rel_pose[self.dim:]
        ang_direction = Geometry.regularize_orient(
            self.k_theta * angle_lin_diff +
            (1.0 - self.k_theta) * angle_ang_diff
            )
        print("angle diff", angle_lin_diff, angle_ang_diff, ang_direction)

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

        print(desired_lin_vel, desired_ang_vel, desired_vel)

        return desired_vel

    def _get_desired_action(self, desired_vel: np.ndarray, vel: np.ndarray, orient: np.ndarray) -> np.ndarray:
        """
        Calculates the action to be taken to reach the desired velocity.

        Parameters:
            desired_vel: Desired velocity in world frame.
            vel: Current velocity in world frame.
            orient: Current orientation in world frame.

        Returns:
            np.ndarray: Action to be taken to reach the desired velocity.
        """
        acc = (desired_vel - vel) / self.dt
        action = FrameTransformer.vel_world_to_robot(self.dim, acc, orient)
        action = self.clip_action(action)
        return action

    def _get_lookahead_pose(self, pos: np.ndarray) -> np.ndarray:
        """
        Get the lookahead pose (x, y, theta) on the path.
        Find the intersection point of the path with a circle centered at the robot's position with radius lookahead_distance.
        If the goal pose is within the lookahead distance, return the goal pose.
        If there are multiple intersections, return the one ahead of the robot along the path.
        If there are no intersections, return the closest pose on the path.

        Parameters:
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
        
        Parameters:
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
                theta = theta1 + t * (theta2 - theta1)
                theta = Geometry.regularize_orient(theta)
                pose = np.array([xy[0], xy[1], theta])
                intersections.append((t, pose))

        return intersections

    def _closest_point_on_path(self, pos: np.ndarray, path: np.ndarray) -> np.ndarray:
        """
        Find the closest point (with theta interpolation) on a polyline path to pos.
        
        Parameters:
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
                proj = p1 + t * d
                theta_proj = Geometry.regularize_orient(path[i, 2] + t * (path[i + 1, 2] - path[i, 2]))

            dist_sq = np.dot(pos - proj, pos - proj)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_pose = np.array([proj[0], proj[1], theta_proj])

        return closest_pose

    




# from typing import List, Tuple
# import numpy as np

# from .base_controller import BaseController

# class PurePursuit(BaseController):
#     """
#     Class of Pure Pursuit path-tracking controller.

#     Parameters:
#         observation_space: observation space ([pos, vel, rel_pos_robot1, rel_pos_robot2, ...], each sub-vector length=dim)
#         action_space: action space ([acc], length=dim)
#         dt: time step for control
#         path: path to follow
#         max_speed: maximum speed of the robot
#         lookahead_distance: lookahead distance for path tracking
#     """
#     def __init__(self,
#                  observation_space,
#                  action_space,
#                  dt: float,
#                  path: List[Tuple[float, ...]] = [],
#                  max_speed: float = np.inf,
#                  lookahead_distance: float = 2.0):
#         super().__init__(observation_space, action_space, dt, path, max_speed)
#         self.lookahead_distance = lookahead_distance
#         self.current_target_index = 0

#     def reset(self):
#         """
#         Reset the controller to initial state.
#         """
#         super().reset()
#         self.current_target_index = 0

#     def get_action(self, obs: np.ndarray) -> Tuple[np.ndarray, tuple]:
#         """
#         Get action from observation.

#         Parameters:
#             obs: observation ([pos, vel, rel_pos_robot1, rel_pos_robot2, ...], each sub-vector length=dim)

#         Returns:
#             action: action ([acc], length=dim)
#             target: lookahead point
#         """
#         if self.goal is None:
#             return np.zeros(self.action_space.shape), self.goal

#         dim = self.action_space.shape[0]
#         pos = obs[:dim]
#         vel = obs[dim:2*dim]

#         # Find the lookahead point
#         target = self._get_lookahead_point(pos)

#         desired_vel = self._get_desired_vel(target, pos)
#         acc = self._get_acc(desired_vel, vel)

#         return acc, target

#     def _get_desired_vel(self, target: np.ndarray, pos: np.ndarray) -> np.ndarray:
#         """
#         Calculate the desired velocity.

#         Parameters:
#             target: the lookahead point
#             pos: the current position

#         Returns:
#             the desired velocity
#         """
#         direction = target - pos
#         distance = np.linalg.norm(direction)
#         if distance > 1e-6:
#             direction /= distance  # Unit Vector

#         # Simple acceleration policy: a = k*(desired_vel - current_vel)
#         max_speed = np.linalg.norm(self.action_space.high)
#         desired_speed = min(distance / self.dt, max_speed)
#         desired_vel = direction * desired_speed
#         desired_vel = self.clip_velocity(desired_vel)

#         return desired_vel

#     def _get_acc(self, desired_vel: np.ndarray, vel: np.ndarray) -> np.ndarray:
#         """
#         Calculates the acceleration vector.

#         Parameters:
#             desired_vel: The desired velocity vector.
#             vel: The current velocity vector.

#         Returns:
#             The acceleration vector.
#         """
#         acc = (desired_vel - vel) / self.dt

#         max_acc = np.min(self.action_space.high)
#         acc_norm = np.linalg.norm(acc)
#         if acc_norm > max_acc:
#             acc = acc / acc_norm * max_acc

#         acc = self.clip_action(acc)
#         return acc

#     def _get_lookahead_point(self, pos: np.ndarray) -> np.ndarray:
#         """
#         Get the lookahead point on the path.
#         Find the intersection point of the path with a circle centered at the robot's position with radius lookahead_distance.
#         If the goal point is within the lookahead distance, return the goal point.
#         If there are multiple intersections, return the one ahead of the robot along the path.
#         If there are no intersections, return the closest point on the path.

#         Parameters:
#             pos: robot position

#         Returns:
#             lookahead_point: lookahead point
#         """
#         # Turn path into numpy array for easier computation
#         path = np.array(self.path)
#         lookahead_sq = self.lookahead_distance **2
#         end_point = path[-1]  # get the goal point
        
#         # Check if the goal point is within the lookahead distance
#         end_dist_sq = np.dot(end_point - pos, end_point - pos)
#         if end_dist_sq <= lookahead_sq + 1e-6:
#             return end_point
        
#         candidates = []
        
#         # Iterate through each segment
#         for i in range(len(path) - 1):
#             # Get the start and end points of the segment
#             p1 = path[i]
#             p2 = path[i + 1]
            
#             d = p2 - p1  # vector of the segment
#             v = pos - p1  # vector from segment start to current position
            
#             # calculate projection length (numerator of line segment parameter t)
#             t_numerator = np.dot(v, d)
#             t_denominator = np.dot(d, d)
            
#             # if line segment length is 0, skip
#             if t_denominator < 1e-10:
#                 continue
                
#             t = t_numerator / t_denominator  # line segment parameter t
            
#             # calculate closest point on the segment
#             if t < 0.0:
#                 closest = p1
#             elif t > 1.0:
#                 closest = p2
#             else:
#                 closest = p1 + t * d
                
#             # calculate distance squared to current position
#             dist_sq = np.dot(pos - closest, pos - closest)
            
#             # if distance is less than or equal to lookahead distance, check for intersection
#             if dist_sq <= lookahead_sq + 1e-6:
#                 # calculate intersection point
#                 if t_denominator < 1e-10:
#                     continue
                    
#                 # solve quadratic equation to get intersection parameter
#                 a = t_denominator
#                 b = -2 * t_numerator
#                 c = np.dot(v, v) - lookahead_sq
#                 discriminant = b**2 - 4*a*c
                
#                 if discriminant < 0:
#                     continue  # no real roots
                
#                 sqrt_d = np.sqrt(discriminant)
#                 t1 = (-b + sqrt_d) / (2*a)
#                 t2 = (-b - sqrt_d) / (2*a)
                
#                 # check if intersection point is on the segment
#                 for t_intersect in [t1, t2]:
#                     if 0.0 <= t_intersect <= 1.0:
#                         intersect_point = p1 + t_intersect * d
#                         # record intersection point and its position in the path
#                         candidates.append((i + t_intersect, intersect_point))
        
#         # if there are intersections, choose the one that is further along the path
#         if candidates:
#             # sort by path position and take the largest one
#             candidates.sort(key=lambda x: x[0])
#             return candidates[-1][1]
        
#         # if no intersections are found, return the closest point on the path
#         min_dist_sq = float('inf')
#         closest_point = path[0]
        
#         for point in path:
#             dist_sq = np.dot(pos - point, pos - point)
#             if dist_sq < min_dist_sq:
#                 min_dist_sq = dist_sq
#                 closest_point = point
                
#         return closest_point
