from typing import List, Tuple
import math

import numpy as np

from python_motion_planning.common.utils.geometry import Geometry
from python_motion_planning.common.utils.frame_transformer import FrameTransformer
from .path_tracker import PathTracker

class PurePursuit(PathTracker):
    """
    Pure Pursuit path-tracking controller.

    Args:
        *args: see the parent class.
        **kwargs: see the parent class.
    """
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

    def _get_desired_vel(self, target_pose: np.ndarray, cur_pose: np.ndarray) -> np.ndarray:
        """
        Calculate the desired velocity in robot frame using pure pursuit.

        Args:
            target_pose: target pose in world frame (from lookahead)
            cur_pose: current pose in world frame

        Returns:
            desired_vel: desired velocity in robot frame [lin_x, lin_y, ang_z] (for 2D)
        """
        # transform target pose into robot frame
        rel_pose = FrameTransformer.pose_world_to_robot(self.dim, target_pose, cur_pose)

        x = rel_pose[0]
        y = rel_pose[1]
        L = math.hypot(x, y)

        # if lookahead distance is (nearly) zero, no movement
        if L < self.eps:
            desired_vel = np.zeros(self.action_space.shape[0])
            return self.clip_velocity(desired_vel)

        # Pure Pursuit curvature: kappa = 2*y / L^2
        # Note: y is lateral offset in robot frame (positive left). For our coordinate,
        # forward x, lateral y. Angular velocity = kappa * v.
        kappa = (2.0 * y) / (L * L)

        desired_lin_speed = self.max_lin_speed * np.sign(x)
        desired_ang_speed = max(min(kappa * desired_lin_speed, self.max_ang_speed), -self.max_ang_speed)

        desired_lin_vel = np.array([desired_lin_speed, 0.0])
        desired_ang_vel = np.array([desired_ang_speed])

        desired_vel = np.concatenate([desired_lin_vel, desired_ang_vel])
        desired_vel = self.clip_velocity(desired_vel)

        return desired_vel
