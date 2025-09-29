"""
@file: circular_robot.py
@breif: CircularRobot class for the environment.
@author: Wu Maojia
@update: 2025.9.20
"""
from typing import List, Dict, Tuple, Optional
import numpy as np


class CircularRobot:
    """
    Base class for circular robots with orientation support.

    Args:
        dim: Space dimension
        mass: Mass of the robot
        radius: Radius of the robot
        pose: Current pose (position + orientation) (world frame)
        vel: Current velocity (linear + angular) (world frame)
        max_lin_speed: Maximum linear speed
        max_ang_speed: Maximum angular speed
        action_min: Minimum action bounds (robot frame)
        action_max: Maximum action bounds (robot frame)
        color: Visualization color
        alpha: Visualization alpha
        fill: Visualization fill
        linewidth: Visualization linewidth
        linestyle: Visualization linestyle
        text: Visualization text (visualized in the center of the robot)
        text_color: Visualization text color
        fontsize: Visualization text fontsize
    """
    def __init__(self, dim: int = 2, mass: float = 1.0, radius: float = 0.5,
                 pose: Optional[np.ndarray] = None, vel: Optional[np.ndarray] = None, 
                 max_lin_speed: float = np.inf, max_ang_speed: float = np.inf,
                 action_min: Optional[np.ndarray] = None, action_max: Optional[np.ndarray] = None,
                 color: str = "C0", alpha: float = 1.0, fill: bool = True, linewidth: float = 1.0, linestyle: str = "-",
                 text: str = "", text_color: str = 'white', fontsize: str = None):
        self.dim = dim
        if dim not in (2, 3):
            raise NotImplementedError(f"Only 2D and 3D are supported, got {dim}D")

        self.pose_dim = 3 if dim == 2 else 6
            
        self.mass = float(mass)
        self.radius = float(radius)
        
        # Pose: position + orientation
        # 2D: [x, y, theta] where theta is angle in radians
        # 3D: [x, y, z, roll, pitch, yaw]
        self.pose = np.zeros(self.pose_dim) if pose is None else np.array(pose, dtype=float)
        self.vel = np.zeros(self.pose_dim) if vel is None else np.array(vel, dtype=float)

        if len(self.pose) != self.pose_dim:
            raise ValueError(f"len(pose) must be {self.pose_dim} if dim=={self.dim}, got {len(pose)}")
        
        if len(self.vel) != self.pose_dim:
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

        # visualization parameters
        self.color = color
        self.alpha = alpha
        self.fill = fill
        self.linewidth = linewidth
        self.linestyle = linestyle
        self.text = text
        self.text_color = text_color
        self.fontsize = fontsize

    @property
    def pos(self):
        """Get position from pose"""
        return self.pose[:self.dim]
    
    @pos.setter
    def pos(self, value):
        """Set position in pose"""
        self.pose[:self.dim] = value

    @property
    def orient(self):
        """Get orientation from pose"""
        return self.pose[self.dim:self.dim*2]

    @orient.setter
    def orient(self, value):
        """Set orientation in pose"""
        self.pose[self.dim:self.dim*2] = value

    @property
    def lin_vel(self):
        """Get linear velocity"""
        return self.vel[:self.dim]

    @lin_vel.setter
    def lin_vel(self, value):
        """Set linear velocity"""
        self.vel[:self.dim] = value

    @property
    def ang_vel(self):
        """Get angular velocity"""
        return self.vel[self.dim:self.dim*2]
    
    @ang_vel.setter
    def ang_vel(self, value):
        """Set angular velocity"""
        self.vel[self.dim:self.dim*2] = value

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
    
    def step(self, env_acc: np.ndarray, dt: float) -> None:
        """
        Take a step in the simulation.

        Args:
            env_acc: acceleration vector from the environment
            dt: time step size
        """
        net_acc = self.acc + env_acc    # self.acc is clipped. env_acc no need to clip.

        # semi-implicit Euler integration
        self.vel = self.vel + net_acc * dt

        # clip linear and angular velocity
        self.vel = self.clip_velocity(self.vel)

        # update pose
        self.pose = self.pose + self.vel * dt

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



    


# from typing import List, Dict, Tuple, Optional

# import numpy as np


# class CircularRobot:
#     """
#     Base class for circular robots.

#     Args:
#         id: Unique robot ID
#         dim: Space dimension
#         mass: Mass of the robot
#         radius: Radius of the robot
#         pos: Current position
#         vel: Current velocity
#         action_min: Minimum action bounds
#         action_max: Maximum action bounds
#         color: Visualization color
#         alpha: Visualization alpha
#         fill: Visualization fill
#         linewidth: Visualization linewidth
#         linestyle: Visualization linestyle
#         text: Visualization text (visualized in the center of the robot)
#         text_color: Visualization text color
#         fontsize: Visualization text fontsize
#     """
#     def __init__(self, id: str = "1", dim: int = 2, mass: float = 1.0, radius: float = 0.1,
#                  pos: Optional[np.ndarray] = None, vel: Optional[np.ndarray] = None, max_speed: float = np.inf,
#                  action_min: Optional[np.ndarray] = None, action_max: Optional[np.ndarray] = None,
#                  color: str = "C0", alpha: float = 1.0, fill: bool = True, linewidth: float = 1.0, linestyle: str = "-",
#                  text: str = "", text_color: str = 'white', fontsize: str = None):
#         self.dim = dim
#         self.mass = float(mass)
#         self.radius = float(radius)
#         self.pos = np.zeros(dim) if pos is None else np.array(pos, dtype=float)
#         self.vel = np.zeros(dim) if vel is None else np.array(vel, dtype=float)
#         self.max_speed = max_speed
#         # acceleration is set externally by controller each step
#         self.acc = np.zeros(dim)
#         # action bounds per-dim (controller output bounds)
#         if action_min is None:
#             action_min = -np.ones(dim) * 1.0
#         if action_max is None:
#             action_max = np.ones(dim) * 1.0
#         self.action_min = np.array(action_min, dtype=float)
#         self.action_max = np.array(action_max, dtype=float)

#         # visualization parameters
#         self.color = color
#         self.alpha = alpha
#         self.fill = fill
#         self.linewidth = linewidth
#         self.linestyle = linestyle
#         self.text = text
#         self.text_color = text_color
#         self.fontsize = fontsize

#     def observation_size(self, env) -> int:
#         """
#         Default observation space: [pos, vel, rel_pos_robot1, rel_pos_robot2, ...], each sub-vector length=dim
#         You can override this function to change the observation structure.

#         Args:
#             env(BaseWorld): World environment

#         Returns:
#             int: Observation size
#         """
#         n_robots = len(env.robots)
#         return 2 * self.dim + (n_robots - 1) * self.dim

#     def get_observation(self, env) -> np.ndarray:
#         """
#         Get observation vector for this robot.

#         Args:
#             env(BaseWorld): World environment

#         Returns:
#             np.ndarray: Observation vector (Default: [pos, vel, rel_pos_robot1, rel_pos_robot2, ...], each sub-vector length=dim)
#         """
#         obs = []
#         obs.extend(self.pos.tolist())
#         obs.extend(self.vel.tolist())
#         for rid, robot in env.robots.items():
#             if robot is self:
#                 continue
#             rel = (robot.pos - self.pos)
#             obs.extend(rel.tolist())
#         return np.array(obs, dtype=float)
    
#     def clip_velocity(self, v: np.ndarray) -> np.ndarray:
#         """
#         Clip the velocity to the maximum allowed value.

#         Args:
#             v (np.ndarray): The velocity to clip.

#         Returns:
#             np.ndarray: The clipped velocity.
#         """
#         return v if np.linalg.norm(v) <= self.max_speed else v / np.linalg.norm(v) * self.max_speed

#     def clip_action(self, a: np.ndarray) -> np.ndarray:
#         """
#         Clip action to action bounds.

#         Args:
#             a(np.ndarray): Action vector

#         Returns:
#             np.ndarray: Clipped action vector
#         """
#         return np.minimum(np.maximum(a, self.action_min), self.action_max)