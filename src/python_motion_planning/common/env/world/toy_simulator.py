from typing import List, Dict, Tuple, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math

from python_motion_planning.common.env.types import TYPES
from python_motion_planning.common.env.map import Grid
from python_motion_planning.common.env.robot import BaseRobot


class ToySimulator(gym.Env):
    """
    多智能体导航环境（可 N 维）
    - robots: list of BaseRobot 的实例
    - bounds: environment boundary as (min_vec, max_vec) each of length dim
    - dt: 时间步长
    - friction: 线性阻尼系数（v * (-friction) force），模型里转成加速度影响
    - restitution: 边界/碰撞弹性系数 [0,1]
    - max_episode_steps: 可选终止步数
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, dim: int = 2,
                 obstacle_grid: Grid = Grid(),
                 dt: float = 0.05,
                 friction: float = 0.1,
                 restitution: float = 0.9,
                 max_episode_steps: int = 1000):
        super().__init__()
        self.dim = dim
        self.obstacle_grid = obstacle_grid
        # self.bounds = (np.array(obstacle_grid.bounds[0], dtype=float), np.array(obstacle_grid.bounds[1], dtype=float))
        self.dt = float(dt)
        self.friction = float(friction)
        self.restitution = float(restitution)
        self.max_episode_steps = int(max_episode_steps)

        self.robots: Dict[str, BaseRobot] = {}
        self.step_count = 0
        # observation_space and action_space are per-robot; environment doesn't expose global spaces
        # 但为了兼容 gym，提供 a dummy space
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(1,), dtype=float)
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(1,), dtype=float)

    def add_robot(self, rid: str, robot: BaseRobot):
        if robot.dim != self.dim:
            raise ValueError("Robot dimension must match environment dimension")
        self.robots[rid] = robot

    def reset(self, seed: Optional[int] = None):
        self.step_count = 0
        # optionally randomize initial states or rely on robots' initial pos/vel
        # 返回 dict of observations keyed by robot index
        obs = {}
        for rid, robot in self.robots.items():
            obs[rid] = robot.get_observation(self)
        return obs, {}

    def step(self, actions: Dict[int, np.ndarray]):
        """
        actions: dict mapping robot_index -> acceleration ndarray (dim,)
        1) clip to robot action bounds
        2) apply environment forces (friction) and integrate via semi-implicit Euler
        3) handle collisions (robot-robot, robot-boundary)
        返回：obs_dict, reward_dict, done_dict, info
        """
        self.step_count += 1
        # 1. assign actions (accelerations) to robots
        for rid, robot in self.robots.items():
            act = actions.get(rid, np.zeros(self.dim))
            robot.acc = robot.clip_action(np.array(act, dtype=float))

        # 2. apply environment forces -> compute net acceleration: robot_net = robot.acc + robot_env (friction)
        for rid, robot in self.robots.items():
            # friction as linear damping: a_fric = -friction * v / mass
            robot_env_acc = - self.friction * robot.vel / (robot.mass + 1e-12)
            robot_net = robot.acc + robot_env_acc
            # semi-implicit Euler: v += robot_net*dt, pos += v*dt
            robot.vel = robot.vel + robot_net * self.dt
            robot.pos = robot.pos + robot.vel * self.dt

        # 3. collisions: pairwise robot-robot elastic collisions (simple impulse) and boundary collisions
        self._resolve_robot_collisions()
        self._resolve_boundary_collisions()

        obs = {rid: robot.get_observation(self) for rid, robot in self.robots.items()}
        # no rewards by default; you can extend
        rewards = {rid: 0.0 for rid in self.robots}
        dones = {rid: False for rid in self.robots}
        terminated = False
        truncated = self.step_count >= self.max_episode_steps
        info = {}
        return obs, rewards, dones, {"terminated": terminated, "truncated": truncated, **info}

    def _resolve_robot_collisions(self):
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
                    rel_vel = np.dot(b.vel - a.vel, nvec)
                    # compute impulse scalar (elastic) with restitution
                    e = self.restitution
                    j_impulse = -(1 + e) * rel_vel / (1 / a.mass + 1 / b.mass)
                    if j_impulse < 0:
                        # apply impulse
                        a.vel -= (j_impulse / a.mass) * nvec
                        b.vel += (j_impulse / b.mass) * nvec
                    # positional correction (simple)
                    overlap = min_dist - dist
                    corr = nvec * (overlap / 2.0 + 1e-6)
                    a.pos = a.pos - corr
                    b.pos = b.pos + corr

    def _resolve_boundary_collisions(self):
        grid = self.obstacle_grid.type_map
        cell_size = self.obstacle_grid.resolution
        for rid, robot in self.robots.items():
            min_pos = (robot.pos[0] - robot.radius, robot.pos[1] - robot.radius)
            max_pos = (robot.pos[0] + robot.radius, robot.pos[1] + robot.radius)
            min_idx = self.obstacle_grid.worldToMap((min_pos[0], min_pos[1]))
            min_idx = tuple(max(0, min_idx[d]) for d in range(self.dim))
            max_idx = self.obstacle_grid.worldToMap((max_pos[0], max_pos[1]))
            max_idx = tuple(min(grid.shape[d] - 1, max_idx[d]) for d in range(self.dim))

            for i in range(min_idx[0], max_idx[0] + 1):
                for j in range(min_idx[1], max_idx[1] + 1):
                    if grid[i, j] == TYPES.OBSTACLE:
                        cell_center = self.obstacle_grid.mapToWorld((i, j))
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
                            vn = robot.vel[0] * nx + robot.vel[1] * ny
                            if vn < 0:  # reflect only if moving towards the cell
                                robot.vel = np.array([robot.vel[0] - (1 + self.restitution) * vn * nx,
                                            robot.vel[1] - (1 + self.restitution) * vn * ny])

    def render(self, mode="human", ax=None):
        # delegated to demo functions; keep signature for Gym compatibility
        raise NotImplementedError("render(): use provided demo_2d/demo_3d functions for visualization.")

    def close(self):
        pass

    # helper to build spaces per-robot
    def build_robot_spaces(self, robot: BaseRobot) -> Tuple[spaces.Box, spaces.Box]:
        """
        返回 (observation_space, action_space) for given robot
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