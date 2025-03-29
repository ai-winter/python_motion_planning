"""
@file: world.py
@breif: Motion Planning Base World
@author: Wu Maojia
@update: 2025.3.29
"""
from typing import Iterable, Union
from abc import ABC, abstractmethod

import numpy as np
import gym
from gym import spaces

class World(gym.Env):
    def __init__(self):
        super().__init__()
        
        # 定义动作空间和观测空间
        self.action_space = spaces.Discrete(4)  # 上、下、左、右
        self.observation_space = spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32)
        
        # 初始化地图和机器人位置
        self.grid_size = 100
        self.robot_pos = np.array([0, 0])
        self.goal_pos = np.array([9, 9])
        self.obstacles = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]
        
        # 可视化设置
        self.fig, self.ax = plt.subplots()
    
    def reset(self):
        # 重置环境状态
        self.robot_pos = np.array([0, 0])
        return self.robot_pos
    
    def step(self, action):
        # 执行动作
        if action == 0:   # 上
            self.robot_pos[1] += 1
        elif action == 1: # 下
            self.robot_pos[1] -= 1
        elif action == 2: # 左
            self.robot_pos[0] -= 1
        elif action == 3: # 右
            self.robot_pos[0] += 1
            
        # 边界检查
        self.robot_pos = np.clip(self.robot_pos, 0, self.grid_size-1)
        
        # 检查是否碰到障碍物
        done = False
        reward = -0.1  # 每一步的小惩罚
        if tuple(self.robot_pos) in self.obstacles:
            reward = -10
            done = True
        elif np.array_equal(self.robot_pos, self.goal_pos):
            reward = 10
            done = True
            
        return self.robot_pos, reward, done, {}
    
    def render(self, mode='human'):
        # 可视化环境
        self.ax.clear()
        
        # 绘制网格
        for x in range(self.grid_size+1):
            self.ax.axvline(x, color='gray', linestyle='-')
            self.ax.axhline(x, color='gray', linestyle='-')
        
        # 绘制障碍物
        for obs in self.obstacles:
            self.ax.add_patch(plt.Rectangle(obs, 1, 1, color='red'))
            
        # 绘制目标
        self.ax.add_patch(plt.Rectangle(self.goal_pos, 1, 1, color='green'))
        
        # 绘制机器人
        self.ax.add_patch(plt.Circle((self.robot_pos[0]+0.5, self.robot_pos[1]+0.5), 0.4, color='blue'))
        
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        self.ax.set_aspect('equal')
        plt.pause(0.1)
    
    def close(self):
        plt.close()


class World(ABC):
    """
    Class for Motion Planning Base World. It is continuous and in n-d Cartesian coordinate system.

    Parameters:
        bounds: boundaries of world (length of boundaries means the number of dimensions)
        dtype: data type of coordinates (must be float)

    Examples:
        >>> world = World((30, 40))

        >>> world
        World((30, 40))

        >>> world.bounds
        (30, 40)

        >>> world.ndim
        2

        >>> world.dtype
        <class 'numpy.float64'>
    """
    def __init__(self, bounds: Iterable, dtype: np.dtype = np.float64) -> None:
        super().__init__()
        try:
            self._bounds = tuple(bounds)
            self._ndim = len(self._bounds)
            self._dtype = dtype

            if self._ndim <= 1:
                raise ValueError("Input length must be greater than 1.")
            if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in self._bounds):
                raise ValueError("Input must be a non-empty 1D array.")

            self._dtype_options = [np.float64, np.float32, np.float16]
            if self._dtype not in self._dtype_options:
                raise ValueError("Dtype must be one of {} instead of {}".format(self._dtype_options, self._dtype))

        except Exception as e:
            raise ValueError("Invalid input for World: {}".format(e))

    def __str__(self) -> str:
        return "World({})".format(self._bounds)

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def bounds(self) -> tuple:
        return self._bounds

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def reset(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError
        


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)