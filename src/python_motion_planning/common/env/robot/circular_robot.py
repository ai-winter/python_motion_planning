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
    Base class for circular robots.

    Parameters:
        id: Unique robot ID
        dim: Space dimension
        mass: Mass of the robot
        radius: Radius of the robot
        pos: Current position
        vel: Current velocity
        action_min: Minimum action bounds
        action_max: Maximum action bounds
        color: Visualization color
        alpha: Visualization alpha
        fill: Visualization fill
        linewidth: Visualization linewidth
        linestyle: Visualization linestyle
        text: Visualization text (visualized in the center of the robot)
        text_color: Visualization text color
        fontsize: Visualization text fontsize
    """
    def __init__(self, id: str = "1", dim: int = 2, mass: float = 1.0, radius: float = 0.1,
                 pos: Optional[np.ndarray] = None, vel: Optional[np.ndarray] = None, max_speed: float = np.inf,
                 action_min: Optional[np.ndarray] = None, action_max: Optional[np.ndarray] = None,
                 color: str = "C0", alpha: float = 1.0, fill: bool = True, linewidth: float = 1.0, linestyle: str = "-",
                 text: str = "", text_color: str = 'white', fontsize: str = None):
        self.dim = dim
        self.mass = float(mass)
        self.radius = float(radius)
        self.pos = np.zeros(dim) if pos is None else np.array(pos, dtype=float)
        self.vel = np.zeros(dim) if vel is None else np.array(vel, dtype=float)
        self.max_speed = max_speed
        # acceleration is set externally by controller each step
        self.acc = np.zeros(dim)
        # action bounds per-dim (controller output bounds)
        if action_min is None:
            action_min = -np.ones(dim) * 1.0
        if action_max is None:
            action_max = np.ones(dim) * 1.0
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

    def observation_size(self, env) -> int:
        """
        Default observation space: [pos, vel, rel_pos_robot1, rel_pos_robot2, ...], each sub-vector length=dim
        You can override this function to change the observation structure.

        Parameters:
            env(BaseWorld): World environment

        Returns:
            int: Observation size
        """
        n_robots = len(env.robots)
        return 2 * self.dim + (n_robots - 1) * self.dim

    def get_observation(self, env) -> np.ndarray:
        """
        Get observation vector for this robot.

        Parameters:
            env(BaseWorld): World environment

        Returns:
            np.ndarray: Observation vector (Default: [pos, vel, rel_pos_robot1, rel_pos_robot2, ...], each sub-vector length=dim)
        """
        obs = []
        obs.extend(self.pos.tolist())
        obs.extend(self.vel.tolist())
        for rid, robot in env.robots.items():
            if robot is self:
                continue
            rel = (robot.pos - self.pos)
            obs.extend(rel.tolist())
        return np.array(obs, dtype=float)
    
    def clip_velocity(self, v: np.ndarray) -> np.ndarray:
        """
        Clip the velocity to the maximum allowed value.

        Parameters:
            v (np.ndarray): The velocity to clip.

        Returns:
            np.ndarray: The clipped velocity.
        """
        return v if np.linalg.norm(v) <= self.max_speed else v / np.linalg.norm(v) * self.max_speed

    def clip_action(self, a: np.ndarray) -> np.ndarray:
        """
        Clip action to action bounds.

        Parameters:
            a(np.ndarray): Action vector

        Returns:
            np.ndarray: Clipped action vector
        """
        return np.minimum(np.maximum(a, self.action_min), self.action_max)