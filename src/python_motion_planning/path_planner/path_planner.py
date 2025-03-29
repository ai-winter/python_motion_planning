"""
@file: planner.py
@breif: Abstract class for planner
@author: Wu Maojia
@update: 2025.3.29
"""
from typing import Union
from abc import ABC, abstractmethod
 
from python_motion_planning.common import Map, PointND

class PathPlanner(ABC):
    """
    Class for building path planner.

    Parameters:
        map_: The map which the planner is based on.
    """
    def __init__(self, map_: Map) -> None:
        super().__init__()
        self.map_ = map_
        self.failed_info = [], {"success": False, "start": None, "goal": None, "length": 0, "cost": 0, "expand": []}

    @abstractmethod
    def plan(self, start: PointND, goal: PointND) -> Union[list, dict]:
        """
        Interface for planning.

        Parameters:
            start: Start point
            goal: Goal point

        Returns:
            path: A list containing the path waypoints
            path_info: A dictionary containing the path information (success, length, cost, expand)
        """
        return self.failed_info
