"""
@file: env.py
@breif: 2-dimension environment
@author: Winter
@update: 2023.1.13
"""
from math import sqrt
from abc import ABC, abstractmethod
from turtle import distance
from scipy.spatial import cKDTree
import numpy as np

from .node import Node

class Env3D(ABC):
    """
    Class for building 3-d workspace of robots.

    Parameters:
        x_range (int): x-axis range of enviroment
        y_range (int): y-axis range of environmet
        z_range (int): z-axis range of environmet
        eps (float): tolerance for float comparison

    Examples:
        >>> from python_motion_planning.utils import Env
        >>> env = Env(30, 40)
    """
    def __init__(self, x_range: int, y_range: int, z_range: int, eps: float = 1e-6) -> None:
        # size of environment
        self.x_range = x_range  
        self.y_range = y_range
        self.z_range = z_range
        self.eps = eps

    @property
    def grid_map(self) -> set:
        return {(i, j, k) for i in range(self.x_range) for j in range(self.y_range) for k in range(self.z_range)}
    
    @abstractmethod
    def init(self) -> None:
        pass

class Grid3D(Env3D):
    """
    Class for discrete 2-d grid map.

    Parameters:
        x_range (int): x-axis range of enviroment
        y_range (int): y-axis range of environmet
    """
    def __init__(self, x_range: int, y_range: int, z_range: int) -> None:
        super().__init__(x_range, y_range, z_range)
        # allowed motions
        self.motions = []

        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                for z in [-1, 0, 1]:
                    if (x, y, z) == (0, 0, 0):
                        continue
                    distance = abs(x) + abs(y) + abs(z)
                    cost = 0
                    if distance == 1:
                        cost = 1
                    elif distance == 2:
                        cost = sqrt(2) 
                    else: 
                        cost = sqrt(3)
                    self.motions.append(Node((x,y,z), None, cost, None))

        # obstacles
        self.obstacles = None
        self.obstacles_tree = None
        self.init()
    
    def init(self) -> None:
        """
        Initialize grid map.
        """
        x, y, z = self.x_range, self.y_range, self.z_range
        obstacles = set()

        # boundary of environment
        for ix in range(x):
            for iy in range(y):
                obstacles.add((ix, iy, 0))
                obstacles.add((ix, iy, z - 1))

        for ix in range(x):
            for iz in range(z):
                obstacles.add((ix, 0, iz))
                obstacles.add((ix, y - 1, iz))

        for iy in range(y):
            for iz in range(z):
                obstacles.add((0, iy, iz))
                obstacles.add((x - 1, iy, iz))

        self.update(obstacles)

    def update(self, obstacles):
        self.obstacles = obstacles 
        self.obstacles_tree = cKDTree(np.array(list(obstacles)))

class Env(ABC):
    """
    Class for building 2-d workspace of robots.

    Parameters:
        x_range (int): x-axis range of enviroment
        y_range (int): y-axis range of environmet
        eps (float): tolerance for float comparison

    Examples:
        >>> from python_motion_planning.utils import Env
        >>> env = Env(30, 40)
    """
    def __init__(self, x_range: int, y_range: int, eps: float = 1e-6) -> None:
        # size of environment
        self.x_range = x_range  
        self.y_range = y_range
        self.eps = eps

    @property
    def grid_map(self) -> set:
        return {(i, j) for i in range(self.x_range) for j in range(self.y_range)}

    @abstractmethod
    def init(self) -> None:
        pass

class Map(Env):
    """
    Class for continuous 2-d map.

    Parameters:
        x_range (int): x-axis range of enviroment
        y_range (int): y-axis range of environmet
    """
    def __init__(self, x_range: int, y_range: int) -> None:
        super().__init__(x_range, y_range)
        self.boundary = None
        self.obs_circ = None
        self.obs_rect = None
        self.init()

    def init(self):
        """
        Initialize map.
        """
        x, y = self.x_range, self.y_range

        # boundary of environment
        self.boundary = [
            [0, 0, 1, y],
            [0, y, x, 1],
            [1, 0, x, 1],
            [x, 1, 1, y]
        ]
        self.obs_rect = []
        self.obs_circ = []

    def update(self, boundary=None, obs_circ=None, obs_rect=None):
        self.boundary = boundary if boundary else self.boundary
        self.obs_circ = obs_circ if obs_circ else self.obs_circ
        self.obs_rect = obs_rect if obs_rect else self.obs_rect
