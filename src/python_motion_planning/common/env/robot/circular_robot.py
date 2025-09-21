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
    Robot 基类：只包含物理参数和状态容器，负责定义观测格式。
    - dim: 空间维度（N）
    - mass: 质量
    - radius: 形状半径（用于碰撞）
    - pos, vel: 当前状态（numpy arrays）
    - acc: 当前瞬时加速度（set by controller before env.step 使用）
    """
    def __init__(self, id: str = "1", dim: int = 2, mass: float = 1.0, radius: float = 0.1,
                 pos: Optional[np.ndarray] = None, vel: Optional[np.ndarray] = None,
                 action_min: Optional[np.ndarray] = None, action_max: Optional[np.ndarray] = None,
                 color: str = "C0", alpha: float = 1.0, fill: bool = True, linewidth: float = 1.0, linestyle: str = "-",
                 text: str = "", text_color: str = 'white', fontsize: str = None):
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
        默认观测：自身 pos, vel (2*dim) + 所有其他 robot 的相对位置 ( (n-1)*dim )
        你可以重载该函数改变观测结构。
        """
        n_robots = len(env.robots)
        return 2 * self.dim + (n_robots - 1) * self.dim

    def get_observation(self, env) -> np.ndarray:
        """
        返回观测向量（1D numpy array）。
        默认格式： [pos, vel, rel_pos_robot1, rel_pos_robot2, ...]
        相对位置按照 env.robots 列表顺序（跳过 self）。
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

    def clip_action(self, a: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(a, self.action_min), self.action_max)