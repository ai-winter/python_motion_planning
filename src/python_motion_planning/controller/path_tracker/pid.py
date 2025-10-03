"""
@file: pid.py
@author: Wu Maojia
@update: 2025.10.3
"""
from typing import List, Tuple
import math

import numpy as np

from python_motion_planning.common.utils.geometry import Geometry
from python_motion_planning.common.utils.frame_transformer import FrameTransformer
from .path_tracker import PathTracker

class PID(PathTracker):
    """
    PID-based path-tracking controller.

    Args:
        *args: see the parent class.
        Kp: proportional gain
        Ki: integral gain
        Kd: derivative gain
        **kwargs: see the parent class.

    References:
        [1] Directional stability of automatically steered bodies
    """
    def __init__(self,
                 *args,
                 Kp: float = 1.0,
                 Ki: float = 0.1,
                 Kd: float = 0.1,
                 **kwargs):
        super().__init__(*args, **kwargs)
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

        Args:
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
