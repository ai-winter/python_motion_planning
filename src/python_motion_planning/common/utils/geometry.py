"""
@file: a_star.py
@breif: A* planner
@author: Wu Maojia
@update: 2025.9.6
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

    # @staticmethod
    # def angle(v1: tuple, v2: tuple) -> float:
    #     """
    #     Calculate the angle between two vectors

    #     Args:
    #         v1: First vector
    #         v2: Second vector

    #     Returns:
    #         angle_rad: Angle in rad between the two vectors
    #     """
    #     if len(v1) != len(v2):
    #         raise ValueError("Dimension mismatch")
        
    #     dot_product = sum(a * b for a, b in zip(v1, v2))
    #     v1_norm = math.sqrt(sum(a**2 for a in v1))
    #     v2_norm = math.sqrt(sum(b**2 for b in v2))
        
    #     if  v1_norm == 0 or v2_norm == 0:
    #         raise ValueError("Zero vector cannot calculate angle")

    #     cos_theta = dot_product / (v1_norm * v2_norm)

    #     cos_theta = min(1.0, max(-1.0, cos_theta))

    #     angle_rad = math.acos(cos_theta)
        
    #     return angle_rad

    @staticmethod
    def regularize_orient(orient: np.ndarray) -> np.ndarray:
        """
        Regularize orientation to be within (-pi, pi]
        """
        return np.mod(orient + np.pi, 2 * np.pi) - np.pi
        
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