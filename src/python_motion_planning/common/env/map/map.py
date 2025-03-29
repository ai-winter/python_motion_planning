"""
@file: map.py
@breif: Map for Path Planning
@author: Wu Maojia
@update: 2025.3.29
"""
from typing import Iterable, Union
from abc import ABC, abstractmethod

import numpy as np

from python_motion_planning.common.env import World, Node
from python_motion_planning.common.geometry.point import *


class Map(ABC):
    """
    Class for Path Planning Map.

    Parameters:
        world: Base world.
        dtype: data type of coordinates
    """
    def __init__(self, world: Union[World, Iterable], dtype: np.dtype) -> None:
        super().__init__()
        if isinstance(world, World):
            self.world = world
        elif isinstance(world, Iterable):
            self.world = World(world)
        else:
            raise ValueError("Invalid world input.")

        self._dtype = dtype

    @property
    def world(self) -> World:
        return self._world
    
    @world.setter
    def world(self, world: World) -> None:
        self._world = world

    @property
    def bounds(self) -> tuple:
        return self.world.bounds

    @property
    def ndim(self) -> int:
        return self.world.ndim

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @abstractmethod
    def mapToWorld(self, point: PointND) -> PointND:
        """
        Convert map coordinates to world coordinates.
        
        Parameters:
            point: Point in map coordinates.
        
        Returns:
            point: Point in world coordinates.
        """
        pass

    @abstractmethod
    def worldToMap(self, point: PointND) -> PointND:
        """
        Convert world coordinates to map coordinates.
        
        Parameters:
            point: Point in world coordinates.
        
        Returns:
            point: Point in map coordinates.
        """
        pass

    @abstractmethod
    def getDistance(self, p1: PointND, p2: PointND) -> float:
        """
        Get the distance between two points.

        Parameters:
            p1: First point.
            p2: Second point.
        
        Returns:
            dist: Distance between two points.
        """
        pass
        
    @abstractmethod
    def getNeighbor(self, node: Node) -> list:
        """
        Get neighbor nodes of a given node.
        
        Parameters:
            node: Node to get neighbor nodes.
        
        Returns:
            nodes: List of neighbor nodes.
        """
        pass

    @abstractmethod
    def inCollision(self, p1: PointND, p2: PointND) -> bool:
        """
        Check if the line of sight between two points is in collision.
        
        Parameters:
            p1: Start point of the line.
            p2: End point of the line.
        
        Returns:
            in_collision: True if the line of sight is in collision, False otherwise.
        """
        pass
        