"""
@file: base_robot.py
@author: Wu Maojia
@update: 2025.10.3
"""
from typing import List, Dict, Tuple, Optional
import numpy as np

from python_motion_planning.common.utils.geometry import Geometry


class BaseRobot:
    """
    Base class for robots.

    Args:
        dim: Space dimension
        mass: Mass of the robot
        pose: Current pose (position + orientation) (world frame)
        vel: Current velocity (linear + angular) (world frame)
        max_lin_speed: Maximum linear speed
        max_ang_speed: Maximum angular speed
        action_min: Minimum action bounds (robot frame)
        action_max: Maximum action bounds (robot frame)
    """
    def __init__(self, dim: int = 2, mass: float = 1.0,
                 pose: Optional[np.ndarray] = None, vel: Optional[np.ndarray] = None, 
                 max_lin_speed: float = np.inf, max_ang_speed: float = np.inf,
                 action_min: Optional[np.ndarray] = None, action_max: Optional[np.ndarray] = None):
        self.dim = dim
        if dim not in (2, 3):
            raise NotImplementedError(f"Only 2D and 3D are supported, got {dim}D")

        self.pose_dim = 3 if dim == 2 else 6
            
        self.mass = float(mass)
        
        # Pose: position + orientation
        # 2D: [x, y, theta] where theta is angle in radians
        # 3D: [x, y, z, roll, pitch, yaw]
        self._pose = np.zeros(self.pose_dim) if pose is None else np.array(pose, dtype=float)
        self._vel = np.zeros(self.pose_dim) if vel is None else np.array(vel, dtype=float)

        if len(self._pose) != self.pose_dim:
            raise ValueError(f"len(pose) must be {self.pose_dim} if dim=={self.dim}, got {len(pose)}")
        
        if len(self._vel) != self.pose_dim:
            raise ValueError(f"len(vel) must be {self.pose_dim} if dim=={self.dim}, got {len(vel)}")

        self.max_lin_speed = max_lin_speed
        self.max_ang_speed = max_ang_speed
        
        # acceleration is set externally by controller each step
        # 2D actions: [longitudinal_acc, lateral_acc, angular_acc]
        # 3D actions: [x_acc, y_acc, z_acc, roll_acc, pitch_acc, yaw_acc]
        self.acc = np.zeros(self.pose_dim)
        
        # action bounds per-dim (controller output bounds)
        if action_min is None:
            action_min = -np.ones(self.pose_dim) * 1.0
        if action_max is None:
            action_max = np.ones(self.pose_dim) * 1.0
        self.action_min = np.array(action_min, dtype=float)
        self.action_max = np.array(action_max, dtype=float)

    @property
    def pose(self):
        """Get position + orientation"""
        return self._pose

    @pose.setter
    def pose(self, value: np.ndarray) -> None:
        """Set position + orientation"""
        self._pose[:self.dim] = value[:self.dim]
        self._pose[self.dim:] = Geometry.regularize_orient(value[self.dim:])

    @property
    def vel(self):
        """Get linear + angular velocity"""
        return self._vel

    @vel.setter
    def vel(self, value: np.ndarray) -> None:
        """Set linear + angular velocity"""
        self._vel = value 

    @property
    def pos(self):
        """Get position from pose"""
        return self._pose[:self.dim]
    
    @pos.setter
    def pos(self, value: np.ndarray):
        """Set position in pose"""
        self._pose[:self.dim] = value

    @property
    def orient(self):
        """Get orientation from pose"""
        return self._pose[self.dim:]

    @orient.setter
    def orient(self, value: np.ndarray):
        """Set orientation in pose"""
        self._pose[self.dim:] = Geometry.regularize_orient(value)

    @property
    def lin_vel(self):
        """Get linear velocity"""
        return self._vel[:self.dim]

    @lin_vel.setter
    def lin_vel(self, value: np.ndarray):
        """Set linear velocity"""
        self._vel[:self.dim] = value

    @property
    def ang_vel(self):
        """Get angular velocity"""
        return self._vel[self.dim:]
    
    @ang_vel.setter
    def ang_vel(self, value: np.ndarray):
        """Set angular velocity"""
        self._vel[self.dim:] = value

    def observation_size(self, env) -> int:
        """
        Default observation space: [pos, orientation, vel, ang_vel]
        """
        # Pose (Position (dim) + orientation (1 if dim==2 or 3 if dim==3)) + 
        #   velocity (linear velocity (dim) + angular velocity (1 if dim==2 or 3 if dim==3))
        if self.dim == 2:   
            orient_dim = 1
        elif self.dim == 3:
            orient_dim = 3
        else:
            raise ValueError("Invalid dimension")

        return 2 * self.dim + 2 * orient_dim

    def get_observation(self, env) -> np.ndarray:
        """
        Get observation vector for this robot including orientation.
        """
        obs = []
        obs.extend(self.pos.tolist())  # Position
        obs.extend(self.orient.tolist())  # Orientation
        obs.extend(self.lin_vel.tolist())  # Linear velocity
        obs.extend(self.ang_vel.tolist())  # Angular velocity
        
        return np.array(obs, dtype=float)

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
        net_acc = acc + env_acc    # acc is clipped. env_acc no need to clip.

        # semi-implicit Euler integration
        vel = vel + net_acc * dt

        # clip linear and angular velocity
        vel = self.clip_velocity(vel)

        # update pose
        pose = pose + vel * dt

        return pose, vel, {}

    def step(self, env_acc: np.ndarray, dt: float) -> None:
        """
        Take a step in simulation using differential drive kinematics.
        self.acc and self.vel are in world frame. You have to transform them into robot frame if needed.

        Args:
            env_acc: acceleration vector from environment
            dt: time step size
        """
        self.pose, self.vel, info = self.kinematic_model(self.pose, self.vel, self.acc, env_acc, dt)

    def clip_linear_velocity(self, lv: np.ndarray) -> np.ndarray:
        """Clip linear velocity to maximum allowed value."""
        return lv if np.linalg.norm(lv) <= self.max_lin_speed else lv / np.linalg.norm(lv) * self.max_lin_speed

    def clip_angular_velocity(self, av: float or np.ndarray) -> float or np.ndarray:
        """Clip angular velocity to maximum allowed value."""
        if self.dim == 2:
            return av if abs(av) <= self.max_ang_speed else np.sign(av) * self.max_ang_speed
        else:  # 3D
            norm = np.linalg.norm(av)
            return av if norm <= self.max_ang_speed else av / norm * self.max_ang_speed

    def clip_velocity(self, v: np.ndarray) -> np.ndarray:
        """Clip linear and angular velocity to maximum allowed value."""
        lv = v[:self.dim]
        av = v[self.dim:self.pose_dim]
        return np.concatenate([self.clip_linear_velocity(lv), self.clip_angular_velocity(av)])

    def clip_action(self, a: np.ndarray) -> np.ndarray:
        """Clip action to action bounds."""
        return np.minimum(np.maximum(a, self.action_min), self.action_max)
