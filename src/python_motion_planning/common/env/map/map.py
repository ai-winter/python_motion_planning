"""
@file: map.py
@breif: Map for Path Planning
@author: Wu Maojia
@update: 2025.3.29
"""
from abc import ABC, abstractmethod

from python_motion_planning.common.env import Env, Node
from python_motion_planning.common.geometry.point import *


class Map(ABC):
    """
    Class for Path Planning Map.

    Parameters:
        env: Base environment.
    """
    def __init__(self, env: Env) -> None:
        super().__init__()
        self.setEnv(env)

    def setEnv(self, env: Env) -> None:
        """
        Set base environment of Map.
        
        Parameters:
            env: Base environment of Map.
        """
        self._env = env

    def getEnv(self) -> Env:
        """
        Get base environment of Map.
        
        Returns:
            env: Base environment of Map.
        """
        return self._env
    
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