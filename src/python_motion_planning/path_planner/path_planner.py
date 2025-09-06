"""
@file: planner.py
@breif: Abstract class for planner
@author: Wu Maojia
@update: 2025.9.5
"""
from typing import Union
from abc import ABC, abstractmethod
 
from python_motion_planning.common import Map

class PathPlanner(ABC):
    """
    Class for building path planner.

    Parameters:
        map_: The map which the planner is based on.
        start: Start point
        goal: Goal point
    """
    def __init__(self, map_: Map, start: tuple, goal: tuple) -> None:
        super().__init__()
        self.map_ = map_
        self.failed_info = [], {"success": False, "start": None, "goal": None, "length": 0, "cost": 0, "expand": []}

    @abstractmethod
    def plan(self) -> Union[list, dict]:
        """
        Interface for planning.

        Returns:
            path: A list containing the path waypoints
            path_info: A dictionary containing the path information (success, length, cost, expand)
        """
        return self.failed_info
