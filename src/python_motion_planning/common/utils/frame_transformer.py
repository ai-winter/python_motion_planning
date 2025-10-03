"""
@file: frame_transformer.py
@author: Wu Maojia
@update: 2025.10.3
"""
from typing import List, Tuple

import numpy as np
from gymnasium import spaces

from .geometry import Geometry


class FrameTransformer:
    """
    Class for transforming between world and robot frames.
    """

    @staticmethod
    def lin_vel_world_to_robot(dim: int, lin_vel: np.ndarray, orient: np.ndarray) -> np.ndarray:
        """
        Transform linear velocity from world frame to robot frame.

        Args:
            dim (int): Space dimension.
            lin_vel (np.ndarray): Linear velocity in world frame [vx, vy] (2D) or [vx, vy, vz] (3D).
            orient (np.ndarray): Orientation of the robot.

        Returns:
            np.ndarray: Linear velocity in robot frame.
        """
        if dim == 2:
            theta = orient[0]
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, s],
                          [-s, c]])
            return R @ lin_vel[:2]
        elif dim == 3:
            # TODO: rotation with quaternion
            raise NotImplementedError("3D velocity transform is not implemented yet.")
        else:
            raise NotImplementedError("Only 2D and 3D cases are supported.")

    @staticmethod
    def vel_world_to_robot(dim: int, vel: np.ndarray, orient: np.ndarray) -> np.ndarray:
        """
        Transform velocity from world frame to robot frame.

        Args:
            dim (int): Space dimension.
            vel (np.ndarray): Velocity (linear and angular) in world frame.
            orient (np.ndarray): Orientation of the robot.

        Returns:
            np.ndarray: Velocity in robot frame.
        """
        lin_vel_world = vel[:dim]
        lin_vel_robot = FrameTransformer.lin_vel_world_to_robot(dim, lin_vel_world, orient)
        return np.concatenate([lin_vel_robot, vel[dim:]])

    @staticmethod
    def lin_vel_robot_to_world(dim: int, lin_vel: np.ndarray, orient: np.ndarray) -> np.ndarray:
        """
        Transform linear velocity from robot frame to world frame.

        Args:
            dim (int): Space dimension.
            lin_vel (np.ndarray): Linear velocity in robot frame.
            orient (np.ndarray): Orientation of the robot.

        Returns:
            np.ndarray: Linear velocity in world frame.
        """
        if dim == 2:
            theta = orient[0]
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s],
                          [s,  c]])
            return R @ lin_vel[:2]
        elif dim == 3:
            # TODO: rotation with quaternion
            raise NotImplementedError("3D velocity transform is not implemented yet.")
        else:
            raise NotImplementedError("Only 2D and 3D cases are supported.")

    @staticmethod
    def vel_robot_to_world(dim: int, vel: np.ndarray, orient: np.ndarray) -> np.ndarray:
        """
        Transform velocity from robot frame to world frame.

        Args:
            dim (int): Space dimension.
            vel (np.ndarray): Velocity (linear and angular) in robot frame.
            orient (np.ndarray): Orientation of the robot.

        Returns:
            np.ndarray: Velocity in world frame.
        """
        lin_vel_robot = vel[:dim]
        lin_vel_world = FrameTransformer.lin_vel_robot_to_world(dim, lin_vel_robot, orient)
        return np.concatenate([lin_vel_world, vel[dim:]])

    @staticmethod
    def pos_world_to_robot(dim: int, pos_world: np.ndarray, robot_pose: np.ndarray) -> np.ndarray:
        """
        Transform a pos from world frame to robot frame.
        
        Args:
            dim (int): Space dimension.
            pos_world (np.ndarray): Pos coordinates in world frame.
            robot_pose (np.ndarray): Robot pose in world frame.
            
        Returns:
            np.ndarray: Pos coordinates in robot frame.
        """
        if dim != 2:
            raise NotImplementedError("Only 2D pos transform is implemented.")
            
        # pose in world frame
        rx, ry, theta = robot_pose
        c, s = np.cos(theta), np.sin(theta)
        
        # translate to the origin
        tx = pos_world[0] - rx
        ty = pos_world[1] - ry
        
        # rotate (use inverse rotation matrix)
        pos_robot = np.array([
            c * tx + s * ty,
            -s * tx + c * ty
        ])
        
        return pos_robot

    @staticmethod
    def pose_world_to_robot(dim: int, pose_world: np.ndarray, robot_pose: np.ndarray) -> np.ndarray:
        """
        Transform a pose (position and orientation) from world frame to robot frame.
        
        Args:
            dim (int): Space dimension.
            pose_world (np.ndarray): Pose in world frame.
            robot_pose (np.ndarray): Robot pose in world frame.
            
        Returns:
            np.ndarray: Pose in robot frame.
        """
        if dim != 2:
            raise NotImplementedError("Only 2D pose transform is implemented.")
            
        # transform position
        position_robot = FrameTransformer.pos_world_to_robot(
            dim, pose_world[:2], robot_pose
        )
        
        # transform orientation
        orientation_robot = pose_world[2] - robot_pose[2]
        orientation_robot = Geometry.regularize_orient(orientation_robot)
        
        return np.concatenate([position_robot, [orientation_robot]])

    @staticmethod
    def pos_robot_to_world(dim: int, pos_robot: np.ndarray, robot_pose: np.ndarray) -> np.ndarray:
        """
        Transform a pos from robot frame to world frame.
        
        Args:
            dim (int): Space dimension.
            pos_robot (np.ndarray): Pos coordinates in robot frame.
            robot_pose (np.ndarray): Robot pose in world frame.
            
        Returns:
            np.ndarray: Pos coordinates in world frame.
        """
        if dim != 2:
            raise NotImplementedError("Only 2D pos transform is implemented.")
            
        # pose in world frame
        rx, ry, theta = robot_pose
        c, s = np.cos(theta), np.sin(theta)
        
        # rotate to the origin
        tx = c * pos_robot[0] - s * pos_robot[1]
        ty = s * pos_robot[0] + c * pos_robot[1]
        
        # translate
        pos_world = np.array([
            tx + rx,
            ty + ry
        ])
        
        return pos_world
    @staticmethod
    def pose_robot_to_world(dim: int, pose_robot: np.ndarray, robot_pose: np.ndarray) -> np.ndarray:
        """
        Transform a pose (position and orientation) from robot frame to world frame.
        
        Args:
            dim (int): Space dimension.
            pose_robot (np.ndarray): Pose in robot frame.
            robot_pose (np.ndarray): Robot pose in world frame.
            
        Returns:
            np.ndarray: Pose in world frame.
        """
        if dim != 2:
            raise NotImplementedError("Only 2D pose transform is implemented.")
            
        # transform position
        position_world = FrameTransformer.pos_robot_to_world(
            dim, pose_robot[:2], robot_pose
        )
        
        # transform orientation
        orientation_world = pose_robot[2] + robot_pose[2]
        orientation_world = Geometry.regularize_orient(orientation_world)
        
        return np.concatenate([position_world, [orientation_world]])
