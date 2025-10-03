"""
@file: toy_simulator.py
@author: Wu Maojia
@update: 2025.10.3
"""
from typing import List, Dict, Tuple, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math

from python_motion_planning.common.utils.frame_transformer import FrameTransformer  
from python_motion_planning.common.env.types import TYPES
from python_motion_planning.common.env.map import Grid
from python_motion_planning.common.env.robot import BaseRobot
from .base_world import BaseWorld


class ToySimulator(BaseWorld):
    """
    Toy Simulator that supports multi-robot navigation in N-dimensions.

    Args:
        dim: dimension of the world (required >= 2)
        dt: the time step size
        obstacle_grid: obstacle grid
        friction: the linear friction coefficient
        restitution: the boundary/collision restitution coefficient [0,1]
        noise: coefficient of random noise (normal distribution) acceleration proportional to speed exerted to robot [0,1]
        max_episode_steps: the maximum number of steps per episode
        robot_collisions: whether to resolve robot collisions
        boundary_collisions: whether to resolve boundary collisions
    """
    def __init__(self, dim: int = 2,
                 dt: float = 0.1,
                 obstacle_grid: Grid = Grid(),
                 friction: float = 0.015,
                 restitution: float = 0.3,
                 noise: float = 0.01,
                 max_episode_steps: int = 1000,
                 robot_collisions: bool = True,
                 boundary_collisions: bool = True):
        super().__init__()
        self.dim = dim
        self.dt = float(dt)
        self.obstacle_grid = obstacle_grid
        self.friction = float(friction)
        self.restitution = float(restitution)
        self.noise = float(noise)
        self.max_episode_steps = int(max_episode_steps)
        self.robot_collisions = robot_collisions
        self.boundary_collisions = boundary_collisions
        self.step_count = 0

    @property
    def time(self) -> float:
        """
        Returns the current accumulated time.
        """
        return self.step_count * self.dt

    def step(self, actions: Dict[int, np.ndarray]):
        """
        Execute one time step in the environment.

        Args:
            actions: dict mapping robot_index -> acceleration ndarray (dim,)
                1) clip to robot action bounds
                2) apply environment forces (friction) and integrate via semi-implicit Euler
                3) handle collisions (robot-robot, robot-boundary)

        Returns:
            obs_dict: dict mapping robot_index -> observation ndarray (dim,)
            reward_dict: dict mapping robot_index -> reward scalar
            done_dict: dict mapping robot_index -> bool
            info: dict
        """
        self.step_count += 1
        # assign actions (accelerations) to robots
        for rid, robot in self.robots.items():
            act = actions.get(rid, np.zeros_like(robot.acc))    # robot frame
            act = robot.clip_action(np.array(act, dtype=float))
            robot.acc = FrameTransformer.vel_robot_to_world(self.dim, act, robot.orient)    # world frame

        # apply environment forces -> compute net acceleration: robot_net_acc = robot.acc + robot_env_acc (friction)
        for rid, robot in self.robots.items():
            # friction as linear damping: a_fric = -friction * v / mass
            env_acc = self.calculate_frictional_acc(robot) + self.generate_env_noise_acc(robot)
            robot.step(env_acc, self.dt)
            
        # collisions: pairwise robot-robot elastic collisions and boundary collisions
        if self.robot_collisions:
            self._resolve_robot_collisions()
        if self.boundary_collisions:
            self._resolve_boundary_collisions()

        # restrict the position within the grid map
        for rid, robot in self.robots.items():
            for d in range(robot.dim):
                robot.pos[d] = min(max(robot.pos[d], self.obstacle_grid.bounds[d, 0]), self.obstacle_grid.bounds[d, 1])

        obs = {rid: robot.get_observation(self) for rid, robot in self.robots.items()}
        rewards = {rid: 0.0 for rid in self.robots}
        dones = {rid: False for rid in self.robots}
        terminated = False
        truncated = self.step_count >= self.max_episode_steps
        info = {}
        return obs, rewards, dones, {"terminated": terminated, "truncated": truncated, **info}

    def calculate_frictional_acc(self, robot: BaseRobot) -> np.ndarray:
        if np.linalg.norm(robot.vel) < 1e-6:
            return np.zeros(robot.pose_dim)
        robot_vel_direction = robot.vel / np.linalg.norm(robot.vel)
        fri_acc = -self.friction * robot_vel_direction * 9.8    # 9.8 is gravitational acceleration
        if np.linalg.norm(fri_acc * self.dt) > np.linalg.norm(robot.vel):
            fri_acc = -robot.vel / self.dt
        return fri_acc

    def generate_env_noise_acc(self, robot: BaseRobot) -> np.ndarray:
        if np.linalg.norm(robot.vel) < 1e-6:
            return np.zeros(robot.pose_dim)
        std = np.abs(robot.vel) * self.noise + 1e-10
        noise_acc = np.random.normal(loc=0.0, scale=std, size=robot.vel.shape)
        return noise_acc

    def _resolve_robot_collisions(self):
        """
        Resolve CircularRobot-CircularRobot collisions.
        """
        rids = list(self.robots.keys())
        n = len(rids)
        for i_ in range(n):
            for j_ in range(i_ + 1, n):
                i = rids[i_]
                j = rids[j_]
                a = self.robots[i]
                b = self.robots[j]
                delta = b.pos - a.pos
                dist = np.linalg.norm(delta)
                min_dist = a.radius + b.radius
                if dist < min_dist:
                    if dist < 1e-8:
                        # numeric edge-case: jitter them slightly
                        delta = np.random.randn(self.dim) * 1e-6
                        dist = np.linalg.norm(delta)

                    # push them apart and compute elastic collision impulse
                    # normal vector
                    nvec = delta / dist
                    # relative velocity along normal
                    rel_lin_vel = np.dot(b.lin_vel - a.lin_vel, nvec)
                    # compute impulse scalar (elastic) with restitution
                    e = self.restitution
                    j_impulse = -(1 + e) * rel_lin_vel / (1 / a.mass + 1 / b.mass)
                    if j_impulse < 0:
                        # apply impulse
                        a.lin_vel -= (j_impulse / a.mass) * nvec
                        b.lin_vel += (j_impulse / b.mass) * nvec
                    # positional correction (simple)
                    overlap = min_dist - dist
                    corr = nvec * (overlap / 2.0 + 1e-6)
                    a.pos = a.pos - corr
                    b.pos = b.pos + corr

    def _resolve_boundary_collisions(self):
        """
        Resolve robot-boundary collisions.
        """
        grid = self.obstacle_grid.type_map
        cell_size = self.obstacle_grid.resolution
        for rid, robot in self.robots.items():
            min_pos = tuple(robot.pos[d] - robot.radius for d in range(self.dim))
            max_pos = tuple(robot.pos[d] + robot.radius for d in range(self.dim))
            min_idx = self.obstacle_grid.world_to_map(tuple(min_pos[d] for d in range(self.dim)))
            min_idx = tuple(max(0, min_idx[d]) for d in range(self.dim))
            max_idx = self.obstacle_grid.world_to_map(tuple(max_pos[d] for d in range(self.dim)))
            max_idx = tuple(min(grid.shape[d] - 1, max_idx[d]) for d in range(self.dim))

            for i in range(min_idx[0], max_idx[0] + 1):
                for j in range(min_idx[1], max_idx[1] + 1):
                    if grid[i, j] == TYPES.OBSTACLE:
                        cell_center = self.obstacle_grid.map_to_world((i, j))
                        cell_min = tuple(cell_center[d] - cell_size * 0.5 for d in range(self.dim))
                        cell_max = tuple(cell_center[d] + cell_size * 0.5 for d in range(self.dim))
                        
                        # closest point in cell
                        closest_x = max(cell_min[0], min(robot.pos[0], cell_max[0]))
                        closest_y = max(cell_min[1], min(robot.pos[1], cell_max[1]))

                        dx = robot.pos[0] - closest_x
                        dy = robot.pos[1] - closest_y
                        dist_sq = dx*dx + dy*dy

                        if dist_sq < robot.radius * robot.radius:
                            dist = dist_sq**0.5 if dist_sq > 1e-8 else 1e-8
                            # normal vector
                            nx, ny = dx / dist, dy / dist
                            # positional correction (simple)
                            overlap = robot.radius - dist
                            robot.pos = np.array([robot.pos[0] + nx * overlap,
                                         robot.pos[1] + ny * overlap])
                            # velocity reflection
                            vn = robot.lin_vel[0] * nx + robot.lin_vel[1] * ny
                            if vn < 0:  # reflect only if moving towards the cell
                                robot.lin_vel = np.array([robot.lin_vel[0] - (1 + self.restitution) * vn * nx,
                                            robot.lin_vel[1] - (1 + self.restitution) * vn * ny])
