"""
@file: geometry.py
@author: Wu Maojia
@update: 2025.10.3
"""
from typing import List, Tuple
import math

import numpy as np

class Geometry:
    """
    Geometry tools class
    """

    @staticmethod
    def dist(p1: tuple, p2: tuple, type: str = 'Euclidean') -> float:
        """
        Calculate the distance between two points

        Args:
            p1: First point
            p2: Second point
            type: Type of distance calculation, either 'Euclidean' or 'Manhattan'

        Returns:
            dist: Distance between the two points
        """
        if len(p1) != len(p2):
            raise ValueError("Dimension mismatch")
        if type == 'Euclidean':
            return math.sqrt(sum((a - b)** 2 for a, b in zip(p1, p2)))
        elif type == 'Manhattan':
            return sum(abs(a - b) for a, b in zip(p1, p2))
        else:
            raise ValueError("Invalid distance type")

    @staticmethod
    def mod_to_2pi(orient: np.ndarray) -> np.ndarray:
        """
        Regularize orientation to be within [0, 2*pi)

        Args:
            orient: the orientation angle

        Returns:
            new_orient: modded orientation
        """
        return np.mod(orient, 2 * np.pi)

    @staticmethod
    def regularize_orient(orient: np.ndarray) -> np.ndarray:
        """
        Regularize orientation to be within (-pi, pi]

        Args:
            orient: the orientation angle

        Returns:
            new_orient: regularized orientation
        """
        return -np.mod(-orient + np.pi, 2 * np.pi) + np.pi
        
    @staticmethod
    def add_orient_to_2d_path(path: List[Tuple[float, float]]) -> List[Tuple[float, float, float]]:
        """
        Add orientation information to a 2D point path. Each point in the path has a third value representing the angle (in radians) between the current point and the next point.

        Args:
            path: a list of 2D points
            
        Returns:
            new_path: a list of 2D poses
        """
        if len(path) < 2:
            return [(x, y, 0.0) for x, y in path]
        
        path_with_orient = []
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            
            dx = x2 - x1
            dy = y2 - y1
            
            orient = math.atan2(dy, dx)
            
            path_with_orient.append((x1, y1, orient))
        
        # last pose
        last_x, last_y = path[-1]
        path_with_orient.append((last_x, last_y, path_with_orient[-1][2]))
        
        return path_with_orient
