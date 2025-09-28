from typing import List, Tuple
import math

import numpy as np

from python_motion_planning.common.utils.geometry import Geometry
from python_motion_planning.common.utils.frame_transformer import FrameTransformer
from .path_tracker import PathTracker

class PID(PathTracker):
    """
    PID-based path-tracking controller.

    Parameters:
        observation_space: observation space ([pos, orient, lin_vel, ang_vel])
        action_space: action space ([lin_acc, ang_acc])
        dt: time step for control
        path: path to follow
        max_lin_speed: maximum linear speed of the robot
        max_ang_speed: maximum angular speed of the robot
        lookahead_distance: lookahead distance for path tracking
        k_theta: weight of theta error
        Kp: proportional gain
        Ki: integral gain
        Kd: derivative gain
    """
    def __init__(self,
                 observation_space,
                 action_space,
                 dt: float,
                 path: List[Tuple[float, ...]] = [],
                 max_lin_speed: float = np.inf, 
                 max_ang_speed: float = np.inf,
                 lookahead_distance: float = 2.0,
                 k_theta: float = 0.7,
                 Kp: float = 1.0,
                 Ki: float = 0.0,
                 Kd: float = 0.1):
        super().__init__(observation_space, action_space, dt,
                         path, max_lin_speed, max_ang_speed,
                         lookahead_distance, k_theta)
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        # integral and previous error for PID
        self.integral_error = np.zeros(self.action_space.shape[0])
        self.prev_error = np.zeros(self.action_space.shape[0])

    def reset(self):
        """
        Reset the controller to initial state.
        """
        super().reset()
        self.integral_error = np.zeros(self.action_space.shape[0])
        self.prev_error = np.zeros(self.action_space.shape[0])

    def _get_desired_action(self, desired_vel: np.ndarray, vel: np.ndarray, orient: np.ndarray) -> np.ndarray:
        """
        Calculates the action to be taken using PID control to reach the desired velocity.

        Parameters:
            desired_vel: Desired velocity in world frame.
            vel: Current velocity in world frame.
            orient: Current orientation in world frame.

        Returns:
            np.ndarray: Action to be taken ([lin_acc, ang_acc]).
        """
        # velocity error
        error = desired_vel - vel

        # PID terms
        self.integral_error += error * self.dt
        derivative_error = (error - self.prev_error) / self.dt

        control = (self.Kp * error +
                   self.Ki * self.integral_error +
                   self.Kd * derivative_error)

        self.prev_error = error

        action = self.clip_action(control)

        return action
