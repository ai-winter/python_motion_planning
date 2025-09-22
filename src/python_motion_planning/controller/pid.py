"""
@file: pid.py
@breif: PID motion planning
@author: Wu Maojia, Yang Haodong
@update: 2025.9.21
"""
import numpy as np
from typing import List, Tuple

from .pure_pursuit import PurePursuit

class PID(PurePursuit):
    """
    Class of PID (Proportional-Integral-Derivative) path-tracking controller.

    Parameters:
        observation_space: observation space ([pos, vel, rel_pos_robot1, rel_pos_robot2, ...], each sub-vector length=dim)
        action_space: action space ([acc], length=dim)
        path: path to follow
        dt: time step for control
        max_speed: maximum speed of the robot
        lookahead_distance: lookahead distance for path tracking
        k_p: proportional gain
        k_i: integral gain
        k_d: derivative gain
    """
    def __init__(self,
                 observation_space,
                 action_space,
                 path: List[Tuple[float, ...]],
                 dt: float,
                 max_speed: float = np.inf,
                 lookahead_distance: float = 2.0,
                 k_p: float = 1.0,
                 k_i: float = 0.1,
                 k_d: float = 0.2):
        super().__init__(observation_space, action_space, path, dt, max_speed, lookahead_distance)
        
        # PID Parameters - support scalar or vector form (set different parameters for different dimensions)
        dim = action_space.shape[0]
        self.k_p = np.full(dim, k_p) if np.isscalar(k_p) else np.array(k_p)
        self.k_i = np.full(dim, k_i) if np.isscalar(k_i) else np.array(k_i)
        self.k_d = np.full(dim, k_d) if np.isscalar(k_d) else np.array(k_d)
        
        # Initialize error terms
        self.reset()

    def reset(self):
        """
        Reset the controller to initial state.
        """
        super().reset()
        dim = self.action_space.shape[0]
        self.integral_error = np.zeros(dim)
        self.prev_error = np.zeros(dim)     # previous error

    def _get_acc(self, desired_vel: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """
        Calculates the acceleration vector.

        Parameters:
            desired_vel: The desired velocity vector.
            vel: The current velocity vector.

        Returns:
            The acceleration vector.
        """
        # Calculate velocity error
        error = desired_vel - vel
        
        # Calculate integral item (accumulate error)
        self.integral_error += error * self.dt
        
        # Calculate derivative item (error change rate)
        derivative_error = (error - self.prev_error) / self.dt
        self.prev_error = error.copy()
        
        # PID control law
        acc = (
            self.k_p * error + 
            self.k_i * self.integral_error + 
            self.k_d * derivative_error
        )
        
        acc = self.clip_action(acc)
        return acc