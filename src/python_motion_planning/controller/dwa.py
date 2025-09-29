from typing import List, Tuple, Optional
import math

import numpy as np

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
        v_reso: resolution of linear velocity sampling
        w_reso: resolution of angular velocity sampling
        predict_time: forward simulation time horizon
        heading_weight: weight for heading term
        velocity_weight: weight for velocity term
        clearance_weight: weight for obstacle clearance term
        **kwargs: see the parent class.
    """
    def __init__(self,
                 *args,
                 robot_model: BaseRobot,
                 obstacle_grid: Grid = None,
                 v_reso: float = 0.1,
                 w_reso: float = np.deg2rad(10.0),
                 predict_time: float = 1.0,
                 heading_weight: float = 0.8,
                 velocity_weight: float = 0.1,
                 clearance_weight: float = 0.1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.robot_model = robot_model
        self.obstacle_grid = obstacle_grid
        self.v_reso = v_reso
        self.w_reso = w_reso
        self.predict_time = predict_time
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

        pose, vel, pos, orient, lin_vel, ang_vel = self.get_pose_velocity(obs)

        # Find the lookahead pose
        target_pose = self._get_lookahead_pose(pos)

        # compute dynamic window
        dw = self._calc_dynamic_window(vel)

        # search best control within dynamic window
        best_u, best_traj = self._evaluate_trajectories(pose, vel, target_pose, dw)

        # compute action (acceleration) to reach best velocity
        desired_vel = best_u
        desired_vel = self._stop_if_reached(desired_vel, pose)
        robot_vel = FrameTransformer.vel_world_to_robot(self.dim, vel, orient)
        action = self._get_desired_action(desired_vel, robot_vel, orient)

        # target pose = last point of best trajectory
        # target_pose = best_traj[-1] if len(best_traj) > 0 else self.goal
        return action, target_pose

    def _calc_dynamic_window(self, vel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate dynamic window [v_min, v_max, w_min, w_max].

        Args:
            vel: current velocity ([vx, vy, omega])

        Returns:
            dw: dynamic window (lin_range, ang_range)
        """
        v = np.linalg.norm(vel[:self.dim])
        w = vel[self.dim:]

        # velocity limits
        v_min, v_max = 0.0, self.max_lin_speed
        w_min, w_max = -self.max_ang_speed, self.max_ang_speed

        return (v_min, v_max, w_min, w_max)

    def _evaluate_trajectories(self, pose: np.ndarray, vel: np.ndarray, target_pose: np.ndarray,
                               dw: Tuple[float, float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate trajectories in dynamic window and choose the best.

        Args:
            pose: current pose ([x, y, theta])
            vel: current velocity ([vx, vy, omega])
            target_pose: target pose ([x, y, theta])
            dw: dynamic window

        Returns:
            best_u: best velocity command [vx, vy, omega]
            best_traj: best simulated trajectory
        """
        v_min, v_max, w_min, w_max = dw
        best_score = -float("inf")
        best_u = np.zeros_like(vel)
        best_traj = []

        for v in np.arange(v_min, v_max, self.v_reso):
            for w in np.arange(w_min, w_max, self.w_reso):
                orient = pose[self.dim:self.dim*2]
                vel_world = FrameTransformer.vel_robot_to_world(self.dim, np.array([v, 0.0, w]), orient)
                traj = self._forward_sim(pose, vel_world)

                # evaluate cost terms
                heading = self._heading_score(traj, target_pose)
                velocity = v / self.max_lin_speed
                clearance = self._clearance_score(traj)

                score = (self.heading_weight * heading +
                         self.velocity_weight * velocity +
                         self.clearance_weight * clearance)

                if score > best_score:
                    best_score = score
                    best_u = np.array([v, 0.0, w])  # [vx, vy, omega]
                    best_traj = traj
        
        return best_u, np.array(best_traj)

    def _forward_sim(self, pose: np.ndarray, vel: np.ndarray) -> List[np.ndarray]:
        """
        Forward simulate trajectory using robot kinematic model.

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
            Heading score.
        """
        last_pose = traj[-1]
        dist = np.linalg.norm(last_pose[:2] - target_pose[:2])
        # orient_dist = np.abs(last_pose[2] - target_pose[2])
        return 1.0 / (dist + 1e-6)

    def _clearance_score(self, traj: List[np.ndarray]) -> float:
        """
        Compute clearance score (min distance to obstacles).
        If no grid map is provided, return 1.0.
        """
        if self.obstacle_grid is None:
            return 1.0

        min_dist = float("inf")
        for p in traj:
            grid_pt = self.obstacle_grid.world_to_map(tuple(p[:2]))
            if not self.obstacle_grid.is_expandable(grid_pt):
                return 0.0
            # update min distance (Euclidean to occupied cells could be added here)
        return np.clip(min_dist / 10.0, 0.0, 1.0)  # normalized
