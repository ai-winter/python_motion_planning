from typing import List, Dict, Tuple, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math

from python_motion_planning.common.env.types import TYPES
from python_motion_planning.common.env.map import Grid
from python_motion_planning.common.env.robot import BaseRobot


class BaseWorld(gym.Env):
    """
    Base class of world (environment contains physical robots).
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.robots: Dict[str, BaseRobot] = {}
        # observation_space and action_space are per-robot; environment doesn't expose global spaces
        # but we provide a dummy space to match gym's requirement
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(1,), dtype=float)
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(1,), dtype=float)

    def add_robot(self, rid: str, robot: BaseRobot):
        """
        Add a robot to the world.

        Args:
            rid: unique robot id
            robot: robot instance
        """
        if robot.dim != self.dim:
            raise ValueError("Robot dimension must match environment dimension")
        self.robots[rid] = robot

    def reset(self, seed: Optional[int] = None):
        """
        Reset the environment.
        
        Args:
            seed: random seed

        Returns:
            observation: Initial observation after reset
        """
        super().reset(seed=seed)
        self.step_count = 0
        # optionally randomize initial states or rely on robots' initial pos/vel
        # return dict of observations keyed by robot index
        obs = {}
        for rid, robot in self.robots.items():
            obs[rid] = robot.get_observation(self)
        return obs, {}

    def step(self, actions: Dict[int, np.ndarray]):
        """
        Execute one time step in the environment.

        Args:
            actions: dict mapping robot_index -> acceleration ndarray (dim,)

        Returns:
            obs_dict: dict mapping robot_index -> observation ndarray (dim,)
            reward_dict: dict mapping robot_index -> reward scalar
            done_dict: dict mapping robot_index -> bool
            info: dict
        """
        self.step_count += 1

        obs = {rid: robot.get_observation(self) for rid, robot in self.robots.items()}
        # no rewards by default; you can extend
        rewards = {rid: 0.0 for rid in self.robots}
        dones = {rid: False for rid in self.robots}
        terminated = False
        truncated = self.step_count >= self.max_episode_steps
        info = {}
        return obs, rewards, dones, {"terminated": terminated, "truncated": truncated, **info}

    def render(self, mode="human", ax=None):
        """
        Render the environment.
        
        Args:
            mode: rendering mode
            ax: matplotlib axis to render to
        """
        # delegated to demo functions; keep signature for Gym compatibility
        raise NotImplementedError("render(): use provided demo_2d/demo_3d functions for visualization.")

    def close(self):
        pass

    # helper to build spaces per-robot
    def build_robot_spaces(self, robot: BaseRobot) -> Tuple[spaces.Box, spaces.Box]:
        """
        Build observation and action spaces for given robot

        Args:
            robot: given robot
        
        Returns:
            observation_space: shape (observation_size,)
            action_space: shape (dim,) bounded by robot.action_min / action_max
        """
        obs_dim = robot.observation_size(self)
        obs_low = -np.inf * np.ones(obs_dim)
        obs_high = np.inf * np.ones(obs_dim)
        obs_space = spaces.Box(obs_low, obs_high, dtype=float)
        act_low = robot.action_min
        act_high = robot.action_max
        act_space = spaces.Box(act_low, act_high, dtype=float)
        return obs_space, act_space

