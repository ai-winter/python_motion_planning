"""
@file: base_controller.py
@author: Wu Maojia
@update: 2025.10.16
"""
from typing import List, Tuple

import numpy as np
from gymnasium import spaces

from python_motion_planning.common.env import Grid, TYPES, BaseRobot
from python_motion_planning.common.utils.geometry import Geometry

class BaseController:
    """
    Base class for controllers.
    - The controller only knows observation_space, action_space

    Args:
        observation_space: observation space ([pos, orient, lin_vel, ang_vel])
        action_space: action space ([lin_acc, ang_acc])
        dt: time step for control
        path: pose path to follow (world frame)
        max_lin_speed: maximum linear speed of the robot
        max_ang_speed: maximum angular speed of the robot
        goal_dist_tol: goal distance tolerance
        goal_orient_tol: goal orient tolerance
        robot_model: robot model for kinematic parameters
        obstacle_grid: occupancy grid map for collision checking
        eps: epsilon (numerical precision)
    """
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Box,
                 dt: float, path: List[Tuple[float, ...]] = [], 
                 max_lin_speed: float = np.inf, max_ang_speed: float = np.inf,
                 goal_dist_tol: float = 0.5, goal_orient_tol: float = np.deg2rad(5),
                 robot_model: BaseRobot = None, obstacle_grid: Grid = None,
                 eps: float = 1e-8):
        self.observation_space = observation_space
        self.action_space = action_space
        self.dt = dt
        self.path = path
        self.max_lin_speed = max_lin_speed
        self.max_ang_speed = max_ang_speed
        self.goal_dist_tol = goal_dist_tol
        self.goal_orient_tol = goal_orient_tol
        self.eps = eps
        
        # Guess dimension from action space
        if self.action_space.shape[0] == 3:
            self.dim = 2
            self.pose_dim = 3
        elif self.action_space.shape[0] == 6:
            self.dim = 3
            self.pose_dim = 6
        else:
            raise NotImplementedError("Action space shape must be 3 (dim=2) or 6 (dim=3). Other dimensions are not supported yet.")

        if len(self.path) > 0:
            if len(self.path[0]) == self.dim:
                self.path = Geometry.add_orient_to_2d_path(self.path)
            self.goal = self.path[-1]
        else:
            self.goal = None
            self.path = False

        self.pred_traj = np.array([])
        
        if robot_model and robot_model.dim != self.dim:
            raise ValueError("Dimension of robot model and controller must be the same")
        self.robot_model = robot_model

        if obstacle_grid and obstacle_grid.dim != self.dim:
            raise ValueError("Dimension of obstacle grid and controller must be the same")
        self.obstacle_grid = obstacle_grid

    def __str__(self) -> str:
        return "Base Controller"

    def reset(self):
        """
        Reset the controller to initial state.
        """
        pass

    def get_action(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get action from observation.

        Args:
            obs: observation ([pos, orient, lin_vel, ang_vel])

        Returns:
            action: action ([lin_acc, ang_acc])
            target_pose: lookahead pose ([pos, orient])
        """
        return np.zeros(self.action_space.shape), self.goal

    def clip_velocity(self, v: np.ndarray) -> np.ndarray:
        """
        Clip the velocity to the maximum allowed value.

        Args:
            v (np.ndarray): The velocity to clip.

        Returns:
            np.ndarray: The clipped velocity.
        """
        lv = v[:self.dim]   # linear velocity
        av = v[self.dim:]   # angular velocity
        lv = lv if np.linalg.norm(lv) <= self.max_lin_speed else lv / np.linalg.norm(lv) * self.max_lin_speed
        av = av if np.linalg.norm(av) <= self.max_ang_speed else av / np.linalg.norm(av) * self.max_ang_speed
        return np.concatenate([lv, av])

    def clip_action(self, a: np.ndarray) -> np.ndarray:
        """
        Clip action to action bounds.

        Args:
            a(np.ndarray): Action vector

        Returns:
            np.ndarray: Clipped action vector
        """
        return np.clip(a, self.action_space.low, self.action_space.high)

    def get_pose_velocity(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get pose and velocity from observation.

        Args:
            obs: observation ([pos, orient, lin_vel, ang_vel])

        Returns:
            pose: pose ([pos, orient])
            vel: velocity ([lin_vel, ang_vel])
            pos: position ([pos_x, pos_y])
            orient: orientation ([orient])
            lin_vel: linear velocity ([lin_vel_x, lin_vel_y])
            ang_vel: angular velocity (ang_vel)
        """
        pose = obs[:self.pose_dim]
        vel = obs[self.pose_dim:self.pose_dim*2]

        pos = pose[:self.dim]
        orient = pose[self.dim:]
        lin_vel = vel[:self.dim]
        ang_vel = vel[self.dim:]

        return pose, vel, pos, orient, lin_vel, ang_vel

    def _stop_if_reached(self, desired_vel: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """
        Stop if the robot has reached the target (within the tolerance).

        Args:
            desired_vel: the desired velocity (robot frame)
            pose: the current pose (world frame)
        
        Returns:
            desired_vel: the desired velocity (robot frame)
        """
        pos = pose[:self.dim]
        orient = pose[self.dim:]
        goal_pos = self.goal[:self.dim]
        goal_orient = self.goal[self.dim:]

        if np.linalg.norm(pos - goal_pos) < self.goal_dist_tol:
            desired_vel[:self.dim] = 0.0

            if np.abs(Geometry.regularize_orient(orient - goal_orient)) < self.goal_orient_tol:
                desired_vel[self.dim:] = 0.0

        return desired_vel

    def _get_dist_to_nearest_obstacle(self, pos: np.ndarray) -> float:
        """
        Get distance to nearest obstacle.
        
        Args:
            pos: position (length = self.dim)
        
        Returns:
            dist: distance to nearest obstacle
        """
        grid_pt = self.obstacle_grid.world_to_map(tuple(pos))
        if not self.obstacle_grid.within_bounds(grid_pt) or self.obstacle_grid.type_map[grid_pt] == TYPES.OBSTACLE:
            dist = 0.0
        else:
            dist = self.obstacle_grid.esdf[grid_pt] * self.obstacle_grid.resolution # using ESDF to compute distance to nearest obstacle

        return dist
