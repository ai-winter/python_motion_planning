"""
@file: planner.py
@breif: Abstract class for planner
@author: Wu Maojia
@update: 2025.9.5
"""
from typing import Union
from abc import ABC, abstractmethod
 
from python_motion_planning.common import BaseMap

class BasePathPlanner(ABC):
    """
    Class for building path planner.

    Args:
        map_: The map which the planner is based on.
        start: The start point of the planner in the map coordinate system.
        goal: The goal point of the planner in the map coordinate system.
    """
    def __init__(self, map_: BaseMap, start: tuple, goal: tuple) -> None:
        super().__init__()
        self.map_ = map_
        self.map_.update_esdf()
        self.start = start
        self.goal = goal
        self.failed_info = [], {"success": False, "start": None, "goal": None, "length": 0, "cost": 0, "expand": {}}

    @abstractmethod
    def plan(self) -> Union[list, dict]:
        """
        Interface for planning.

        Returns:
            path: A list containing the path waypoints
            path_info: A dictionary containing the path information (success, length, cost, expand)
        """
        return self.failed_info

    def get_cost(self, p1: tuple, p2: tuple) -> float:
        """
        Get the cost between two points. (default: distance defined in the map)

        Args:
            p1: Start point.
            p2: Goal point.
        
        Returns:
            cost: Cost between two points.
        """
        return self.map_.get_distance(p1, p2)
    
    def get_heuristic(self, point: tuple) -> float:
        """
        Get the heuristic value of the point. (default: cost between current point and goal point)

        Args:
            point: Point.

        
        Returns:
            heuristic: Heuristic value of the point.
        """
        return self.get_cost(point, self.goal)

    
    def extract_path(self, closed_list: dict) -> tuple:
        """
        Extract the path based on the CLOSED list.

        Args:
            closed_list (dict): CLOSED list

        Returns:
            cost (float): the cost of planned path
            path (list): the planning path
        """
        length = 0
        cost = 0
        node = closed_list[self.goal]
        path = [node.current]
        while node.current != self.start:
            node_parent = closed_list[node.parent]
            length += self.map_.get_distance(node.current, node_parent.current)
            cost += self.get_cost(node.current, node_parent.current)
            node = node_parent
            path.append(node.current)
        path = path[::-1]   # make the order: start -> goal
        return path, length, cost