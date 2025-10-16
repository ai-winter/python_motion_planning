"""
@file: rpp.py
@author: Wu Maojia
@update: 2025.10.16
"""
from typing import List, Tuple
import math

import numpy as np

from python_motion_planning.common.utils.geometry import Geometry
from python_motion_planning.common.utils.frame_transformer import FrameTransformer
from .pure_pursuit import PurePursuit


class RPP(PurePursuit):
    """
    Regulated Pure Pursuit (RPP) path-tracking controller. `obstacle_grid` must be provided.

    Args:
        *args: see the parent class.
        curvature_thresh (float): minimum curvature threshold T_kappa to trigger speed scaling.
        r_min (float): minimum radius (used in curvature heuristic).
        d_prox (float): proximity distance threshold to start slowing down near obstacles.
        alpha (float): proximity scaling gain (0 < alpha <= 1).
        lookahead_gain (float): lookahead time gain for adaptive lookahead.
        **kwargs: see the parent class.
    """

    def __init__(self,
                 *args,
                 curvature_thresh: float = 0.1,
                 r_min: float = 1.0,
                 d_prox: float = None,
                 alpha: float = 0.8,
                 lookahead_gain: float = 1.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        if self.obstacle_grid is None:
            raise ValueError("Obstacle grid is required.")

        self.curvature_thresh = curvature_thresh
        self.r_min = r_min
        self.d_prox = d_prox if d_prox is not None else self.obstacle_grid.inflation_radius
        self.alpha = alpha
        self.lookahead_gain = lookahead_gain

    def __str__(self) -> str:
        return "RPP"

    def _get_desired_vel(self, target_pose: np.ndarray, cur_pose: np.ndarray) -> np.ndarray:
        """
        Calculate the desired velocity in robot frame using Regulated Pure Pursuit.

        Args:
            target_pose: target pose in world frame (from lookahead)
            cur_pose: current pose in world frame
            obs_dist: distance to nearest obstacle (if available), defaults to infinity

        Returns:
            desired_vel: desired velocity in robot frame [lin_x, lin_y, ang_z]
        """
        # transform target pose to robot frame
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

        # Base linear speed (before regulation)
        v_t = self.max_lin_speed

        # Curvature heuristic
        # Slow down in sharp turns: Eq. (5) in the paper
        if abs(kappa) > self.curvature_thresh:
            v_curv = v_t / (self.r_min * max(abs(kappa), self.eps))
        else:
            v_curv = v_t

        # Proximity heuristic
        # Slow down when close to obstacles: Eq. (6) in the paper
        obs_dist = self._get_dist_to_nearest_obstacle(cur_pose[:self.dim])
        if obs_dist <= self.d_prox:
            v_prox = v_t * self.alpha * obs_dist / max(self.d_prox, self.eps)
        else:
            v_prox = v_t
        v_prox = v_t

        # Take minimum regulated velocity (for safety)
        v_reg = min(v_curv, v_prox)

        # Compute angular velocity using regulated linear speed
        omega = np.clip(v_reg * kappa, -self.max_ang_speed, self.max_ang_speed)

        # Assemble desired velocity vector
        desired_lin_vel = np.array([v_reg * (1.0 if x >= 0 else -1.0), 0.0])
        desired_ang_vel = np.array([omega])
        desired_vel = np.concatenate([desired_lin_vel, desired_ang_vel])

        desired_vel = self.clip_velocity(desired_vel)
        return desired_vel
