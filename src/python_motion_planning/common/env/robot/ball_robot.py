"""
@file: circular_agent.py
@breif: CircularAgent class for the environment.
@author: Wu Maojia
@update: 2025.9.20
"""
from typing import List, Dict, Tuple, Optional

import numpy as np


class BallRobot:
    """
    Agent 基类：只包含物理参数和状态容器，负责定义观测格式。
    - dim: 空间维度（N）
    - mass: 质量
    - radius: 形状半径（用于碰撞）
    - pos, vel: 当前状态（numpy arrays）
    - acc: 当前瞬时加速度（set by controller before env.step 使用）
    """
    def __init__(self, dim: int = 2, mass: float = 1.0, radius: float = 0.1,
                 pos: Optional[np.ndarray] = None, vel: Optional[np.ndarray] = None,
                 action_min: Optional[np.ndarray] = None, action_max: Optional[np.ndarray] = None):
        self.dim = dim
        self.mass = float(mass)
        self.radius = float(radius)
        self.pos = np.zeros(dim) if pos is None else np.array(pos, dtype=float)
        self.vel = np.zeros(dim) if vel is None else np.array(vel, dtype=float)
        # acceleration is set externally by controller each step
        self.acc = np.zeros(dim)
        # action bounds per-dim (controller output bounds)
        if action_min is None:
            action_min = -np.ones(dim) * 1.0
        if action_max is None:
            action_max = np.ones(dim) * 1.0
        self.action_min = np.array(action_min, dtype=float)
        self.action_max = np.array(action_max, dtype=float)

    def observation_size(self, env) -> int:
        """
        默认观测：自身 pos, vel (2*dim) + 所有其他 agent 的相对位置 ( (n-1)*dim )
        你可以重载该函数改变观测结构。
        """
        n_agents = len(env.agents)
        return 2 * self.dim + (n_agents - 1) * self.dim

    def get_observation(self, env) -> np.ndarray:
        """
        返回观测向量（1D numpy array）。
        默认格式： [pos, vel, rel_pos_agent1, rel_pos_agent2, ...]
        相对位置按照 env.agents 列表顺序（跳过 self）。
        """
        obs = []
        obs.extend(self.pos.tolist())
        obs.extend(self.vel.tolist())
        for a in env.agents:
            if a is self:
                continue
            rel = (a.pos - self.pos)
            obs.extend(rel.tolist())
        return np.array(obs, dtype=float)

    def clip_action(self, a: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(a, self.action_min), self.action_max)