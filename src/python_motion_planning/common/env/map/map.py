"""
@file: map.py
@breif: Map for Path Planning
@author: Wu Maojia
@update: 2025.3.29
"""
from typing import Iterable, Union
from abc import ABC, abstractmethod

import numpy as np

from python_motion_planning.common.env import Node
from python_motion_planning.common.geometry.point import *


class Map(ABC):
    """
    Class for Path Planning Map.

    Parameters:
        bounds: The size of map in the world (shape: (n, 2) (n>=2)). bounds[i, 0] means the lower bound of the world in the i-th dimension. bounds[i, 1] means the upper bound of the world in the i-th dimension.  
        dtype: data type of coordinates
    """
    def __init__(self, bounds: Iterable, dtype: np.dtype) -> None:
        super().__init__()
        self._bounds = np.array(bounds, dtype=np.float64)
        self._dtype = dtype

        if len(self._bounds.shape) != 2 or self._bounds.shape[0] <= 1 or self._bounds.shape[1] != 2:
            raise ValueError(f"The shape of bounds must be (n, 2) (n>=2) instead of {self._bounds.shape}")

        for dim in range(self._bounds.shape[0]):
            if self._bounds[dim, 0] >= self._bounds[dim, 1]:
                raise ValueError(f"The lower bound of the world in the {dim}-th dimension must be smaller than the upper bound of the world in the {dim}-th dimension.")

    @property
    def bounds(self) -> np.ndarray:
        return self._bounds

    @property
    def ndim(self) -> int:
        return self._bounds.shape[0]

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
        