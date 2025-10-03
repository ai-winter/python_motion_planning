"""
@file: map.py
@author: Wu Maojia
@update: 2025.10.3
"""
from typing import Iterable, Union
from abc import ABC, abstractmethod

import numpy as np

from python_motion_planning.common.env import Node


class BaseMap(ABC):
    """
    Base class for Path Planning Map.

    Args:
        bounds: The size of map in the world (shape: (n, 2) (n>=2)). bounds[i, 0] means the lower bound of the world in the i-th dimension. bounds[i, 1] means the upper bound of the world in the i-th dimension.  
        dtype: data type of coordinates
    """
    def __init__(self, bounds: Iterable, dtype: np.dtype) -> None:
        super().__init__()
        self._bounds = np.asarray(bounds, dtype=float)
        self._dtype = dtype

        if len(self._bounds.shape) != 2 or self._bounds.shape[0] <= 1 or self._bounds.shape[1] != 2:
            raise ValueError(f"The shape of bounds must be (n, 2) (n>=2) instead of {self._bounds.shape}")

        for d in range(self._bounds.shape[0]):
            if self._bounds[d, 0] >= self._bounds[d, 1]:
                raise ValueError(f"The lower bound of the world in the {d}-th dimension must be smaller than the upper bound of the world in the {d}-th dimension.")

    @property
    def bounds(self) -> np.ndarray:
        return self._bounds

    @property
    def dim(self) -> int:
        return self._bounds.shape[0]

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @abstractmethod
    def map_to_world(self, point: tuple) -> tuple:
        """
        Convert map coordinates to world coordinates.
        
        Args:
            point: Point in map coordinates.
        
        Returns:
            point: Point in world coordinates.
        """
        pass

    @abstractmethod
    def world_to_map(self, point: tuple) -> tuple:
        """
        Convert world coordinates to map coordinates.
        
        Args:
            point: Point in world coordinates.
        
        Returns:
            point: Point in map coordinates.
        """
        pass

    @abstractmethod
    def get_distance(self, p1: tuple, p2: tuple) -> float:
        """
        Get the distance between two points.

        Args:
            p1: First point.
            p2: Second point.
        
        Returns:
            dist: Distance between two points.
        """
        pass
        
    @abstractmethod
    def get_neighbors(self, node: Node) -> list:
        """
        Get neighbor nodes of a given node.
        
        Args:
            node: Node to get neighbor nodes.
        
        Returns:
            nodes: List of neighbor nodes.
        """
        pass

    @abstractmethod
    def is_expandable(self, point: tuple) -> bool:
        """
        Check if a point is expandable.
        
        Args:
            point: Point to check.
        
        Returns:
            expandable: True if the point is expandable, False otherwise.
        """
        pass

    @abstractmethod
    def in_collision(self, p1: tuple, p2: tuple) -> bool:
        """
        Check if the line of sight between two points is in collision.
        
        Args:
            p1: Start point of the line.
            p2: End point of the line.
        
        Returns:
            in_collision: True if the line of sight is in collision, False otherwise.
        """
        pass
        