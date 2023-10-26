'''
@file: env.py
@breif: 2-dimension environment
@author: Winter
@update: 2023.1.13
'''
from math import sqrt
from abc import ABC, abstractmethod
from scipy.spatial import cKDTree
import numpy as np

from .node import Node

class Env(ABC):
    '''
    Class for building 2-d workspace of robots.

    Parameters
    ----------
    x_range: int
        x-axis range of enviroment
    y_range: int
        y-axis range of environmet

    Examples
    ----------
    >>> from utils import Env
    >>> env = Env(30, 40)
    '''
    def __init__(self, x_range: int, y_range: int) -> None:
        # size of environment
        self.x_range = x_range  
        self.y_range = y_range

    @property
    def grid_map(self) -> set:
        return {(i, j) for i in range(self.x_range) for j in range(self.y_range)}

    @abstractmethod
    def init(self) -> None:
        pass

class Grid(Env):
    '''
    Class for discrete 2-d grid map.
    '''
    def __init__(self, x_range: int, y_range: int) -> None:
        super().__init__(x_range, y_range)
        # allowed motions
        self.motions = [Node((-1, 0), None, 1, None), Node((-1, 1),  None, sqrt(2), None),
                        Node((0, 1),  None, 1, None), Node((1, 1),   None, sqrt(2), None),
                        Node((1, 0),  None, 1, None), Node((1, -1),  None, sqrt(2), None),
                        Node((0, -1), None, 1, None), Node((-1, -1), None, sqrt(2), None)]
        # obstacles
        self.obstacles = None
        self.obstacles_tree = None
        self.init()
    
    def init(self) -> None:
        '''
        Initialize grid map.
        '''
        x, y = self.x_range, self.y_range
        obstacles = set()

        # boundary of environment
        for i in range(x):
            obstacles.add((i, 0))
            obstacles.add((i, y - 1))
        for i in range(y):
            obstacles.add((0, i))
            obstacles.add((x - 1, i))

        # user-defined obstacles
        # for i in range(3, 15):
        #     obstacles.add((i, 10))
        # for i in range(7, 11):
        #     obstacles.add((3, i))
        # for i in range(4, 16):
        #     obstacles.add((i, 2))
        # for i in range(3, 7):
        #     obstacles.add((7, i))
        # for i in range(19, 27):
        #     obstacles.add((i, 12))
        # for i in range(2, 7):
        #     obstacles.add((20, i))
        # obstacles.add((21, 2))
        # obstacles.add((22, 2))
        # for i in range(14, 18):
        #     obstacles.add((5, i))
        #     obstacles.add((13, i))
        # for i in range(5, 9):
        #     obstacles.add((25, i))
        #     obstacles.add((26, i))
        # for i in range(15, 18):
        #     obstacles.add((23, i))
        #     obstacles.add((24, i))
        
        for i in range(10, 21):
            obstacles.add((i, 15))
        for i in range(15):
            obstacles.add((20, i))
        for i in range(15, 30):
            obstacles.add((30, i))
        for i in range(16):
            obstacles.add((40, i))

        self.obstacles = obstacles
        self.obstacles_tree = cKDTree(np.array(list(obstacles)))

    def update(self, obstacles):
        self.obstacles = obstacles 
        self.obstacles_tree = cKDTree(np.array(list(obstacles)))


class Map(Env):
    '''
    Class for continuous 2-d map.
    '''
    def __init__(self, x_range: int, y_range: int) -> None:
        super().__init__(x_range, y_range)
        self.boundary = None
        self.obs_circ = None
        self.obs_rect = None
        self.init()

    def init(self):
        '''
        Initialize map.
        '''
        x, y = self.x_range, self.y_range

        # boundary of environment
        self.boundary = [
            [0, 0, 1, y],
            [0, y, x, 1],
            [1, 0, x, 1],
            [x, 1, 1, y]
        ]

        # user-defined obstacles
        self.obs_rect = [
            [14, 12, 8, 2],
            [18, 22, 8, 3],
            [26, 7, 2, 12],
            [32, 14, 10, 2]
        ]

        self.obs_circ = [
            [7, 12, 3],
            [46, 20, 2],
            [15, 5, 2],
            [37, 7, 3],
            [37, 23, 3]
        ]

    def update(self, boundary, obs_circ, obs_rect):
        self.boundary = boundary if boundary else self.boundary
        self.obs_circ = obs_circ if obs_circ else self.obs_circ
        self.obs_rect = obs_rect if obs_rect else self.obs_rect
