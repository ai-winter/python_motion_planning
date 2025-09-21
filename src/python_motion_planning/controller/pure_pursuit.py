from typing import List, Tuple
import numpy as np

from .base_controller import BaseController

class PurePursuit(BaseController):
    """
    Class of Pure Pursuit path-tracking controller.

    Parameters:
        observation_space: observation space ([pos, vel, rel_pos_robot1, rel_pos_robot2, ...], each sub-vector length=dim)
        action_space: action space ([acc], length=dim)
        path: path to follow
        dt: time step for control
        max_speed: maximum speed of the robot
        lookahead_distance: lookahead distance for path tracking
    """
    def __init__(self,
                 observation_space,
                 action_space,
                 path: List[Tuple[float, ...]],
                 dt: float,
                 max_speed: float = np.inf,
                 lookahead_distance: float = 2.0):
        super().__init__(observation_space, action_space, path, dt, max_speed)
        self.lookahead_distance = lookahead_distance
        self.current_target_index = 0

    def reset(self):
        """
        Reset the controller to initial state.
        """
        super().reset()
        self.current_target_index = 0

    def get_action(self, obs: np.ndarray) -> Tuple[np.ndarray, tuple]:
        """
        Get action from observation.

        Parameters:
            obs: observation ([pos, vel, rel_pos_robot1, rel_pos_robot2, ...], each sub-vector length=dim)

        Returns:
            action: action ([acc], length=dim)
            target: lookahead point
        """
        if self.goal is None:
            return np.zeros(self.action_space.shape), self.goal

        dim = self.action_space.shape[0]
        pos = obs[:dim]
        vel = obs[dim:2*dim]

        # Find the lookahead point
        target = self._get_lookahead_point(pos)

        desired_vel = self._get_desired_vel(target, pos)
        acc = self._get_acc(desired_vel, vel)

        return acc, target

    def _get_desired_vel(self, target: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """
        Calculate the desired velocity.

        Parameters:
            target: the lookahead point
            pos: the current position

        Returns:
            the desired velocity
        """
        direction = target - pos
        distance = np.linalg.norm(direction)
        if distance > 1e-6:
            direction /= distance  # Unit Vector

        # Simple acceleration policy: a = k*(desired_vel - current_vel)
        max_speed = np.linalg.norm(self.action_space.high)
        desired_speed = min(distance / self.dt, max_speed)
        desired_vel = direction * desired_speed
        desired_vel = self.clip_velocity(desired_vel)

        return desired_vel

    def _get_acc(self, desired_vel: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """
        Calculates the acceleration vector.

        Parameters:
            desired_vel: The desired velocity vector.
            vel: The current velocity vector.

        Returns:
            The acceleration vector.
        """
        acc = (desired_vel - vel) / self.dt

        max_acc = np.min(self.action_space.high)
        acc_norm = np.linalg.norm(acc)
        if acc_norm > max_acc:
            acc = acc / acc_norm * max_acc

        acc = self.clip_action(acc)
        return acc

    def _get_lookahead_point(self, pos: np.ndarray) -> np.ndarray:
        """
        Get the lookahead point on the path.
        Find the intersection point of the path with a circle centered at the robot's position with radius lookahead_distance.
        If the goal point is within the lookahead distance, return the goal point.
        If there are multiple intersections, return the one ahead of the robot along the path.
        If there are no intersections, return the closest point on the path.

        Parameters:
            pos: robot position

        Returns:
            lookahead_point: lookahead point
        """
        # Turn path into numpy array for easier computation
        path = np.array(self.path)
        lookahead_sq = self.lookahead_distance **2
        end_point = path[-1]  # get the goal point
        
        # Check if the goal point is within the lookahead distance
        end_dist_sq = np.dot(end_point - pos, end_point - pos)
        if end_dist_sq <= lookahead_sq + 1e-6:
            return end_point
        
        candidates = []
        
        # Iterate through each segment
        for i in range(len(path) - 1):
            # Get the start and end points of the segment
            p1 = path[i]
            p2 = path[i + 1]
            
            d = p2 - p1  # vector of the segment
            v = pos - p1  # vector from segment start to current position
            
            # calculate projection length (numerator of line segment parameter t)
            t_numerator = np.dot(v, d)
            t_denominator = np.dot(d, d)
            
            # if line segment length is 0, skip
            if t_denominator < 1e-10:
                continue
                
            t = t_numerator / t_denominator  # line segment parameter t
            
            # calculate closest point on the segment
            if t < 0.0:
                closest = p1
            elif t > 1.0:
                closest = p2
            else:
                closest = p1 + t * d
                
            # calculate distance squared to current position
            dist_sq = np.dot(pos - closest, pos - closest)
            
            # if distance is less than or equal to lookahead distance, check for intersection
            if dist_sq <= lookahead_sq + 1e-6:
                # 计算交点  # calculate intersection point
                if t_denominator < 1e-10:
                    continue
                    
                # solve quadratic equation to get intersection parameter
                a = t_denominator
                b = -2 * t_numerator
                c = np.dot(v, v) - lookahead_sq
                discriminant = b**2 - 4*a*c
                
                if discriminant < 0:
                    continue  # no real roots
                
                sqrt_d = np.sqrt(discriminant)
                t1 = (-b + sqrt_d) / (2*a)
                t2 = (-b - sqrt_d) / (2*a)
                
                # check if intersection point is on the segment
                for t_intersect in [t1, t2]:
                    if 0.0 <= t_intersect <= 1.0:
                        intersect_point = p1 + t_intersect * d
                        # record intersection point and its position in the path
                        candidates.append((i + t_intersect, intersect_point))
        
        # if there are intersections, choose the one that is further along the path
        if candidates:
            # sort by path position and take the largest one
            candidates.sort(key=lambda x: x[0])
            return candidates[-1][1]
        
        # if no intersections are found, return the closest point on the path
        min_dist_sq = float('inf')
        closest_point = path[0]
        
        for point in path:
            dist_sq = np.dot(pos - point, pos - point)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_point = point
                
        return closest_point
