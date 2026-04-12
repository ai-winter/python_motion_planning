"""
@file: base_curve_generator.py
@author: Yang Haodong, Wu Maojia
@update: 2026.4.12
"""
from typing import List, Tuple, Dict, Any
from abc import ABC, abstractmethod
import math


class BaseCurveGenerator(ABC):
    """
    Base class for curve generator (trajectory smoother).

    Args:
        step: Step size for interpolation or discretization along the curve.
    """
    def __init__(self, step: float = 0.01) -> None:
        super().__init__()
        self.step = step

    def __str__(self) -> str:
        return "Base Curve Generator"

    @abstractmethod
    def generate(self, points: List[Tuple[float, ...]]) -> Tuple[List[Tuple[float, ...]], Dict[str, Any]]:
        """
        Interface for curve generation.

        Args:
            points: A list of waypoints in world frame. The exact format (2D position
                or 2D pose with orientation) depends on the concrete generator.

        Returns:
            path: A list containing the smoothed path waypoints in world frame.
            curve_info: A dictionary containing the curve information (success, length).
        """
        raise NotImplementedError

    def length(self, path: List[Tuple[float, ...]]) -> float:
        """
        Calculate the length of a path.

        Args:
            path: A list containing the path waypoints.

        Returns:
            length: Length of the path.
        """
        dist = 0.0
        for i in range(len(path) - 1):
            dist += math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        return dist
