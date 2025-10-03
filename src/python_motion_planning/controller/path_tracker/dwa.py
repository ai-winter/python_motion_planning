"""
@file: dwa.py
@author: Wu Maojia
@update: 2025.10.3
"""
from typing import List, Tuple, Optional
import math

import numpy as np

from python_motion_planning.common.env.types import TYPES
from python_motion_planning.common.env.map.grid import Grid
from python_motion_planning.common.env.robot.base_robot import BaseRobot
from python_motion_planning.common.utils.geometry import Geometry
from python_motion_planning.common.utils.frame_transformer import FrameTransformer
from .path_tracker import PathTracker


class DWA(PathTracker):
    """
    Dynamic Window Approach (DWA) path-tracking controller.

    Args:
        *args: see the parent class.
        robot_model: robot model for kinematic simulation
        obstacle_grid: occupancy grid map for collision checking
        vel_reso: resolution of velocity sampling
        predict_time: forward simulation time horizon
        heading_weight: weight for heading term
        velocity_weight: weight for velocity term
        clearance_weight: weight for obstacle clearance term
        **kwargs: see the parent class.

    References:
        [1] The Dynamic Window Approach to Collision Avoidance
    """
    def __init__(self,
                 *args,
                 robot_model: BaseRobot,
                 obstacle_grid: Grid = None,
                 vel_reso: float = np.array([0.2, 0.2, np.deg2rad(15)]),
                 predict_time: float = None,
                 heading_weight: float = 0.5,
                 velocity_weight: float = 0.2,
                 clearance_weight: float = 0.3,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if robot_model.dim != self.dim:
            raise ValueError("Dimension of robot model and controller must be the same")
        self.robot_model = robot_model

        if obstacle_grid.dim != self.dim:
            raise ValueError("Dimension of obstacle grid and controller must be the same")
        self.obstacle_grid = obstacle_grid

        self.vel_reso = vel_reso
        self.predict_time = predict_time if predict_time is not None else self.lookahead_distance / self.max_lin_speed
        self.heading_weight = heading_weight
        self.velocity_weight = velocity_weight
        self.clearance_weight = clearance_weight

    def get_action(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get action from observation using Dynamic Window Approach.

        Args:
            obs: observation ([pos, orient, lin_vel, ang_vel])

        Returns:
            action: action in robot frame ([lin_acc, ang_acc])
            target_pose: selected local goal ([pos, orient])
        """
        if self.goal is None:
            return np.zeros(self.action_space.shape), self.goal

        pose, vel, pos, orient, lin_vel, ang_vel = self.get_pose_velocity(obs)  # world frame

        # Find the lookahead pose
        target_pose = self._get_lookahead_pose(pos)

        # search best control within dynamic window
        best_vel, best_traj = self._evaluate_trajectories(pose, vel, target_pose)

        # compute action (acceleration) to reach best velocity
        desired_vel = best_vel    # robot frame
        desired_vel = self._stop_if_reached(desired_vel, pose)  # robot frame
        robot_vel = FrameTransformer.vel_world_to_robot(self.dim, vel, orient)
        action = self._get_desired_action(desired_vel, robot_vel, orient)   # robot frame

        self.pred_traj = best_traj  # for visualization

        return action, target_pose

    def _evaluate_trajectories(self, pose: np.ndarray, vel: np.ndarray, target_pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate trajectories in dynamic window and choose the best.

        Args:
            pose: current pose ([x, y, theta])  world frame
            vel: current velocity ([vx, vy, omega]) world frame
            target_pose: target pose ([x, y, theta])    world frame

        Returns:
            best_vel: best velocity command [vx, vy, omega]   robot frame
            best_traj: best simulated trajectory
        """
        # v_min, v_max, w_min, w_max = dw
        best_score = -float("inf")
        best_vel = np.zeros_like(vel)   # should be in world frame
        best_scores = (0.0,0.0,0.0)
        best_traj = []

        orient = pose[self.dim:]
        vel_robot = FrameTransformer.vel_world_to_robot(self.dim, vel, orient)  # robot frame

        vel_points = []
        for d in range(self.action_space.shape[0]):
            low = vel_robot[d] + self.action_space.low[d] * self.dt
            high = vel_robot[d] + self.action_space.high[d] * self.dt
            reso = self.vel_reso[d]

            sample_points = np.arange(low, high, reso)
            
            # make sure the high boundary velocity is included
            sample_points = np.append(sample_points, [high])

            if low < 0 and high > 0:
                sample_points = np.append(sample_points, [0.0])

            sample_points = np.unique(np.round(sample_points / self.eps) * self.eps)  # unique with numerical precision

            vel_points.append(sample_points)
        
        vel_grid = np.meshgrid(*vel_points)
        sampled_vels = np.stack(vel_grid, axis=-1).reshape(-1, vel_robot.shape[0])  # robot frame

        for sampled_vel in sampled_vels:
            # exclude velocities that exceed max speed limitation
            sampled_lin_speed = np.linalg.norm(sampled_vel[:self.dim])
            if sampled_lin_speed > self.max_lin_speed: 
                continue

            sampled_ang_speed = np.linalg.norm(sampled_vel[self.dim:])
            if sampled_ang_speed > self.max_ang_speed: 
                continue

            vel_world = FrameTransformer.vel_robot_to_world(self.dim, sampled_vel, orient)  # world frame
            traj = self._forward_sim(pose, vel_world)

            # evaluate cost terms
            heading = self._heading_score(traj, target_pose)
            velocity = self._velocity_score(sampled_lin_speed)
            clearance = self._clearance_score(traj)

            score = (self.heading_weight * heading +
                        self.velocity_weight * velocity +
                        self.clearance_weight * clearance)

            if score > best_score:
                best_score = score
                best_scores = (self.heading_weight * heading, self.velocity_weight * velocity, self.clearance_weight * clearance)
                best_vel = sampled_vel
                best_traj = traj
        
        return best_vel, np.array(best_traj)

    def _forward_sim(self, pose: np.ndarray, vel: np.ndarray) -> List[np.ndarray]:
        """
        Forward simulate trajectory using robot kinematic model.

        Args:
            pose: pose of robot (world frame)
            vel: velocity of robot (world frame)

        Returns:
            traj: simulated trajectory (list of poses)
        """
        traj = [pose.copy()]
        time = 0.0
        while time <= self.predict_time:
            pose, vel, _ = self.robot_model.kinematic_model(pose, vel, np.zeros_like(vel), np.zeros_like(vel), self.dt)
            traj.append(pose)
            time += self.dt
        return traj

    def _heading_score(self, traj: List[np.ndarray], target_pose: np.ndarray) -> float:
        """
        Compute heading cost (distance to goal).

        Args:
            traj: Trajectory.
            target_pose: Target pose.

        Returns:
            float: Heading score.
        """
        last_pose = traj[-1]
        dist = np.linalg.norm(last_pose[:self.dim] - target_pose[:self.dim])
        orient = np.linalg.norm(Geometry.regularize_orient(last_pose[self.dim:] - target_pose[self.dim:]))

        normalized_dist = dist / self.goal_dist_tol
        normalized_orient = orient / self.goal_orient_tol

        dist_score = 0.5 * np.clip(1.0 / normalized_dist, 0.0, 1.0)     # normalized_dist > 1.0
        dist_score += 0.5 * np.clip(1.0 - normalized_dist, 0.0, 1.0)    # 0.0 <= normalized_dist <= 1.0
        
        orient_score = 0.5 * np.clip(1.0 / normalized_orient, 0.0, 1.0)  # normalized_orient > 1.0
        orient_score += 0.5 * np.clip(1.0 - normalized_orient, 0.0, 1.0) # 0.0 <= normalized_orient <= 1.0

        total_score = self.k_theta * dist_score + (1 - self.k_theta) * orient_score
        
        return total_score

    def _velocity_score(self, lin_speed: np.ndarray) -> float:
        """
        Compute the velocity score.

        Args:
            lin_speed (np.ndarray): The linear speed (robot frame)

        Returns:
            float: The velocity score
        """
        return lin_speed / self.max_lin_speed

    def _clearance_score(self, traj: List[np.ndarray]) -> float:
        """
        Compute clearance score (min distance to obstacles).
        If no grid map is provided, return 1.0.

        Args:
            traj: The trajectory

        Returns:
            float: The clearance score
        """
        if self.obstacle_grid is None:
            return 1.0

        radius = self.robot_model.radius

        min_dist = float("inf") 
        for p in traj:
            grid_pt = self.obstacle_grid.world_to_map(tuple(p[:2]))
            if not self.obstacle_grid.within_bounds(grid_pt) or self.obstacle_grid.type_map[grid_pt] == TYPES.OBSTACLE:
                return -float("inf")     # collision
            # update min distance (Euclidean to occupied cells could be added here)
            dist = self.obstacle_grid.esdf[grid_pt] * self.obstacle_grid.resolution # using ESDF to compute distance to nearest obstacle
            min_dist = min(min_dist, dist)
            
        normalized_min_dist = min_dist / self.robot_model.radius

        if normalized_min_dist < self.eps:
            return -float("inf")

        return np.clip(1.0 - 1.0 / normalized_min_dist, 0.0, 1.0)  # normalized
