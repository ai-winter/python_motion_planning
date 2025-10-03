"""
@file: apf.py
@author: Wu Maojia, Yang Haodong 
@update: 2025.10.3
"""
from typing import List, Tuple
import math

import numpy as np

from python_motion_planning.common.env.types import TYPES
from python_motion_planning.common.env.map.grid import Grid
from python_motion_planning.common.env.robot.base_robot import BaseRobot
from python_motion_planning.common.utils.geometry import Geometry
from python_motion_planning.common.utils.frame_transformer import FrameTransformer
from .path_tracker import PathTracker


class APF(PathTracker):
    """
    Artificial Potential Field (APF) path-tracking controller.

    Args:
        *args: see the parent class.
        robot_model: robot model for kinematic parameters
        obstacle_grid: occupancy grid map for collision checking
        attr_weight: weight factor for attractive potential
        rep_weight: weight factor for repulsive potential
        rep_range: influence range for repulsive potential
        **kwargs: see the parent class.
    """
    def __init__(self,
                 *args,
                 robot_model: BaseRobot,
                 obstacle_grid: Grid = None,
                 attr_weight: float = 1.0,
                 rep_weight: float = 1.0,
                 rep_range: float = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if robot_model.dim != self.dim:
            raise ValueError("Dimension of robot model and controller must be the same")
        self.robot_model = robot_model

        if obstacle_grid and obstacle_grid.dim != self.dim:
            raise ValueError("Dimension of obstacle grid and controller must be the same")
        self.obstacle_grid = obstacle_grid

        # APF parameters
        self.attr_weight = attr_weight  # Attractive potential weight
        self.rep_weight = rep_weight    # Repulsive potential weight
        self.rep_range = rep_range if rep_range is not None else self.robot_model.radius * 2.0  # Repulsive influence range

    def get_action(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get action from observation using Artificial Potential Field method.

        Args:
            obs: observation ([pos, orient, lin_vel, ang_vel])

        Returns:
            action: action in robot frame ([lin_acc, ang_acc])
            target_pose: lookahead pose ([pos, orient])
        """
        if self.goal is None:
            return np.zeros(self.action_space.shape), self.goal

        pose, vel, pos, orient, lin_vel, ang_vel = self.get_pose_velocity(obs)

        # Find the lookahead pose as the target for attractive potential
        target_pose = self._get_lookahead_pose(pos)

        # Calculate potential field forces
        attractive_force = self._calculate_attractive_force(pos, target_pose[:self.dim])
        repulsive_force = self._calculate_repulsive_force(pos)

        # Total force is the sum of attractive and repulsive forces
        total_force = attractive_force + repulsive_force

        # Calculate desired velocity from total force
        desired_vel = self._force_to_velocity(total_force, orient)
        desired_vel = self._stop_if_reached(desired_vel, pose)

        # Convert current velocity to robot frame and calculate action
        robot_vel = FrameTransformer.vel_world_to_robot(self.dim, vel, orient)
        action = self._get_desired_action(desired_vel, robot_vel, orient)

        return action, target_pose

    def _calculate_attractive_force(self, current_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        """
        Calculate attractive force towards the target position.

        Args:
            current_pos: Current position of the robot in world frame
            target_pos: Target position (lookahead point) in world frame

        Returns:
            attractive_force: Attractive force vector in world frame
        """
        # Vector from current position to target position
        direction = target_pos - current_pos
        distance = np.linalg.norm(direction)

        # If distance is very small, return zero force to avoid division issues
        if distance < self.eps:
            return np.zeros_like(direction)

        # Normalize direction and scale by weight and distance
        # For standard APF, attractive force is proportional to distance
        attractive_force = self.attr_weight * direction

        return attractive_force

    def _calculate_repulsive_force(self, current_pos: np.ndarray) -> np.ndarray:
        """
        Calculate repulsive force from obstacles using ESDF.

        Args:
            current_pos: Current position of the robot in world frame

        Returns:
            repulsive_force: Repulsive force vector in world frame
        """
        if self.obstacle_grid is None:
            return np.zeros(self.dim)

        # Convert world position to grid coordinates
        grid_pt = self.obstacle_grid.world_to_map(tuple(current_pos[:2]))
        grid_x, grid_y = grid_pt

        # Check if position is out of bounds or in an obstacle
        if not self.obstacle_grid.within_bounds(grid_pt) or self.obstacle_grid.type_map[grid_pt] == TYPES.OBSTACLE:
            # Large repulsive force if in collision
            return np.full(self.dim, self.rep_weight * self.rep_range)

        # Get distance to nearest obstacle from ESDF (in world units)
        dist_to_obstacle = self.obstacle_grid.esdf[grid_pt] * self.obstacle_grid.resolution

        # No repulsive force if outside influence range
        if dist_to_obstacle >= self.rep_range:
            return np.zeros(self.dim)

        # Calculate gradient of repulsive potential using numpy's gradient function
        # Extract a small window around current grid point to compute gradient
        window_size = 3  # 3x3 window for gradient calculation
        half_window = window_size // 2
        
        # Initialize window with current distance value
        window = np.zeros((window_size, window_size))
        window[half_window, half_window] = self.obstacle_grid.esdf[grid_pt]
        
        # Fill window with neighboring ESDF values, handling boundary conditions
        for i in range(window_size):
            for j in range(window_size):
                # Calculate grid coordinates for this window position
                grid_i = grid_x + (i - half_window)
                grid_j = grid_y + (j - half_window)
                
                # Check if neighboring grid point is within bounds
                if self.obstacle_grid.within_bounds((grid_i, grid_j)):
                    window[i, j] = self.obstacle_grid.esdf[(grid_i, grid_j)]
                else:
                    # For points outside bounds, use distance decreasing away from map
                    dx = abs(i - half_window)
                    dy = abs(j - half_window)
                    dist_from_center = math.sqrt(dx*dx + dy*dy)
                    window[i, j] = max(0, self.obstacle_grid.esdf[grid_pt] - dist_from_center)

        window *= self.obstacle_grid.resolution

        # Calculate gradient using numpy.gradient
        # The gradient is scaled by grid resolution to get world coordinates    
        # TODO: Check if this is correct
        # gy, gx = np.gradient(window) 
        gx, gy = np.gradient(window)
        
        # The gradient at the center of the window gives the direction of maximum increase
        # This is the direction away from obstacles
        gradient_dir = np.array([gx[half_window, half_window], gy[half_window, half_window]])
        
        # Normalize gradient direction
        grad_mag = np.linalg.norm(gradient_dir)
        if grad_mag < 1e-6:
            return np.zeros(self.dim)
        gradient_dir = gradient_dir / grad_mag

        # Calculate repulsive force magnitude using standard APF formula
        rep_force_magnitude = self.rep_weight * (1.0 / dist_to_obstacle - 1.0 / self.rep_range) / (dist_to_obstacle ** 2)

        # Scale direction by repulsive force magnitude
        repulsive_force = rep_force_magnitude * gradient_dir

        return repulsive_force

    def _force_to_velocity(self, force: np.ndarray, orient: np.ndarray) -> np.ndarray:
        """
        Convert force vector to desired velocity in robot frame.

        Args:
            force: Total force vector in world frame
            orient: Current orientation in world frame

        Returns:
            desired_vel: Desired velocity in robot frame
        """
        force_magnitude = np.linalg.norm(force)
        
        # If force is very small, return zero velocity
        if force_magnitude < self.eps:
            return np.zeros(self.action_space.shape[0])

        # Desired linear velocity is proportional to force magnitude
        desired_lin_speed = min(force_magnitude, self.max_lin_speed)
        desired_lin_dir = force / force_magnitude  # Direction of force

        # Desired angular velocity is based on angle difference between force direction and robot orientation
        force_angle = np.array([math.atan2(desired_lin_dir[1], desired_lin_dir[0])])
        angle_diff = Geometry.regularize_orient(force_angle - orient)
        desired_ang_speed = min(np.linalg.norm(angle_diff) / self.dt, self.max_ang_speed)
        desired_ang_speed *= np.sign(angle_diff[0])  # Preserve direction

        desired_ang_speed = np.array([desired_ang_speed])

        # Combine linear and angular velocity
        desired_vel_world = np.concatenate([desired_lin_dir * desired_lin_speed, desired_ang_speed])
        
        # Convert desired velocity to robot frame
        desired_vel_robot = FrameTransformer.vel_world_to_robot(self.dim, desired_vel_world, orient)

        return self.clip_velocity(desired_vel_robot)
    