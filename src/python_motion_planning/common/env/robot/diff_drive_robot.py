"""
@file: diff_drive_robot.py
@author: Wu Maojia
@update: 2025.10.3
"""
from typing import List, Dict, Tuple, Optional

import numpy as np

from python_motion_planning.common.utils.frame_transformer import FrameTransformer
from .circular_robot import CircularRobot


class DiffDriveRobot(CircularRobot):
    """
    Differential drive robot with non-holonomic constraints.
    Inherits from CircularRobot and overrides the dynamics.
    action space should be [longitudinal_vel, 0.0, angular_vel] (lateral velocity should be 0.0)

    Args:
        wheel_base: Distance between left and right wheels
        wheel_radius: Radius of wheels (for conversion between wheel speed and linear velocity)
        Other parameters are the same as CircularRobot.
    """
    def __init__(self, wheel_base: float = 0.5, wheel_radius: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        if self.dim != 2:
            raise NotImplementedError("DiffDriveRobot only supports 2D")

        # diff-drive robot does not have lateral control
        self.action_min[1] = 0.0
        self.action_max[1] = 0.0

        self.wheel_base = float(wheel_base)
        self.wheel_radius = float(wheel_radius)

        # store wheel velocities [left, right]
        self.wheel_vel = np.zeros(2)    # robot frame

    def kinematic_model(self, pose: np.ndarray, vel: np.ndarray, acc: np.ndarray, env_acc: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Kinematic model (used to simulate the robot motion without updating the robot state)

        Args:
            pose: robot pose    (world frame)
            vel: robot velocity (world frame)
            acc: robot acceleration action (world frame)
            env_acc: environment acceleration (world frame)
            dt: time step length

        Returns:
            pose: new robot pose    (world frame)
            vel: new robot velocity (world frame)
            info: auxiliary information
        """
        orient = pose[self.dim:self.dim*2]

        # compute wheel velocities
        action = FrameTransformer.vel_world_to_robot(self.dim, acc, orient) # robot frame

        # compute desired velocities after integration
        vel_robot = FrameTransformer.vel_world_to_robot(self.dim, vel, orient)    # robot frame
        vel_robot[0] += action[0] * dt     # longitudinal velocity robot frame
        # no lateral velocity
        vel_robot[2] += action[2] * dt     # angular velocity robot frame

        # map to wheel velocities
        v_l = vel_robot[0] - (self.wheel_base / 2.0) * vel_robot[2]
        v_r = vel_robot[0] + (self.wheel_base / 2.0) * vel_robot[2]

        wheel_vel = np.array([v_l, v_r])  # robot frame

        # differential drive kinematics
        v = (v_r + v_l) / 2.0       # robot frame
        omega = (v_r - v_l) / self.wheel_base   # robot frame
        
        old_vel_robot = FrameTransformer.vel_world_to_robot(self.dim, vel, orient)    # robot frame
        vel_robot = np.array([v, old_vel_robot[1], omega])   # robot frame

        # lateral motion is restricted by sliding friction
        env_acc_robot = FrameTransformer.vel_world_to_robot(self.dim, env_acc, orient)    # robot frame
        env_acc_robot[1] *= 100.0   # The estimated multiple of sliding friction coefficient to rolling friction coefficient
        if env_acc_robot[1] * dt > vel_robot[1]:
            env_acc_robot[1] = -vel_robot[1] / dt

        vel_robot += env_acc_robot * dt    # environment force

        vel = FrameTransformer.vel_robot_to_world(self.dim, vel_robot, orient)    # world frame

        vel = self.clip_velocity(vel)

        # update pose
        pose = pose + vel * dt

        info = {"wheel_vel": wheel_vel}

        return pose, vel, info

    def step(self, env_acc: np.ndarray, dt: float) -> None:
        """
        Take a step in simulation using differential drive kinematics.
        self.acc and self.vel are in world frame. You have to transform them into robot frame if needed.

        Args:
            env_acc: acceleration vector from environment
            dt: time step size
        """
        self.pose, self.vel, info = self.kinematic_model(self.pose, self.vel, self.acc, env_acc, dt)
        self.wheel_vel = info["wheel_vel"]
