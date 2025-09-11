"""
@file: env.py
@breif: 2-dimension environment
@author: Winter
@update: 2023.1.13
"""
from math import sqrt
from abc import ABC, abstractmethod
from scipy.spatial import cKDTree
import numpy as np

from .node import Node

class Env(ABC):
    """
    Class for building 2-d workspace of robots.

    Parameters:
        x_range (int): x-axis range of environment
        y_range (int): y-axis range of environment
        z_range (int): z-axis range of environment
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
        return {(i, j) for i in range(self.x_range) for j in range(self.y_range)}

    @abstractmethod
    def init(self) -> None:
        pass

class Grid(Env):
    """
    Class for discrete 2-d grid map.

    Parameters:
        x_range (int): x-axis range of enviroment
        y_range (int): y-axis range of enviroment
        z_range (int): z-axis range of enviroment
    """
    def __init__(self, x_range: int, y_range: int, z_range: int) -> None:
        super().__init__(x_range, y_range, z_range)
        # allowed motions
        self.motions = [
                        # Singular direction
                        Node((1, 0, 0), None, 1, None),  # Right
                        Node((-1, 0, 0), None, 1, None), # Left
                        Node((0, 1, 0), None, 1, None),  # Up
                        Node((0, -1, 0), None, 1, None), # Down
                        Node((0, 0, 1), None, 1, None),  # Forward
                        Node((0, 0, -1), None, 1, None), # Backward
                        # XY-Plane
                        Node((1, 1, 0), None, sqrt(2), None), # Right Up
                        Node((-1, 1, 0), None, sqrt(2), None), # Left Up
                        Node((1, -1, 0), None, sqrt(2), None), # Right Down
                        Node((-1, 1, 0), None, sqrt(2), None), # Left Down
                        # XZ-Plane
                        Node((1, 0, 1), None, sqrt(2), None), # Right Forward
                        Node((1, 0, -1), None, sqrt(2), None), # Right Backward
                        Node((-1, 0, 1), None, sqrt(2), None), # Left Forward
                        Node((-1, 0, -1), None, sqrt(2), None), # Left Backward
                        # YZ-Plane
                        Node((0, 1, 1), None, sqrt(2), None), # Up Forward
                        Node((0, 1, -1), None, sqrt(2), None), # Up Backward
                        Node((0, -1, 1), None, sqrt(2), None), # Down Forward
                        Node((0, -1, -1), None, sqrt(2), None), # Down Backward
                        # XYZ-Plane
                        Node((1, 1, 1), None, sqrt(3), None),   # Right Up Forward
                        Node((-1, 1, 1), None, sqrt(3), None),  # Left Up Forward
                        Node((1, 1, -1), None, sqrt(3), None),  # Right Up Backward
                        Node((-1, 1, -1), None, sqrt(3), None), # Left Up Backward
                        Node((1, -1, 1), None, sqrt(3), None), # Right Down Forward
                        Node((-1, -1, 1), None, sqrt(3), None), # Left Down Forward
                        Node((1, -1, -1), None, sqrt(3), None), # Right Down Backward
                        Node((-1, -1, -1), None, sqrt(3), None)  # Left Down Backward
                        ]
        # obstacles
        self.obstacles = None
        self.obstacles_tree = None
        self.inner_obstacles = None
        self.init()

    def init(self) -> None:
        """
        Initialize grid map.
        """
        x, y, z = self.x_range, self.y_range, self.z_range
        obstacles = set()

        # boundary of environment

        # walls
        for i in range(z):
            for j in range(x):
                obstacles.add((j, 0, i))
                obstacles.add((j, y - 1, i))
            for j in range(y):
                obstacles.add((0, j, i))
                obstacles.add((x - 1, j, i))

        # floor and roof
        for i in range(x):
            for j in range(y):
                obstacles.add((i, j, 0))
                obstacles.add((i, j, z - 1))

        self.update(obstacles)

    def update(self, obstacles):
        self.obstacles = obstacles
        self.obstacles_tree = cKDTree(np.array(list(obstacles)))


class Map(Env):
    """
    Class for continuous 2-d map.

    Parameters:
        x_range (int): x-axis range of enviroment
        y_range (int): y-axis range of environmet
    """
    def __init__(self, x_range: int, y_range: int, z_range: int) -> None:
        super().__init__(x_range, y_range, z_range)
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
