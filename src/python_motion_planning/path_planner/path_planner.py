"""
@file: planner.py
@breif: Abstract class for planner
@author: Winter
@update: 2023.1.17
"""
import math

from python_motion_planning.common.geometry import Point3d
from python_motion_planning.common.structure import Node, Env
from python_motion_planning.common.geometry import CollisionChecker

class PathPlanner:
    """
    Class for building path planner.

    Parameters:
        params (dict): parameters
    """
    def __init__(self, env: Env, params: dict) -> None:
        # total parameters
        self.params = params
        for k, v in params["strategy"]["path_planner"].items():
            setattr(self, k, v)
        # environment
        self.env = env
        # obstacles
        self.obstacles = self.env.obstacles
        # collision checker
        self.collision_checker = CollisionChecker(self.env.grid_map)

    def dist(self, node1: Node, node2: Node) -> float:
        return math.hypot(node2.x - node1.x, node2.y - node1.y)
    
    def angle(self, node1: Node, node2: Node) -> float:
        return math.atan2(node2.y - node1.y, node2.x - node1.x)

    def cost(self, node1: Node, node2: Node) -> float:
        """
        Calculate cost for this motion.

        Parameters:
            node1 (Node): node 1
            node2 (Node): node 2

        Returns:
            cost (float): cost of this motion
        """
        if self.collision_checker(node1.current, node2.current, "onestep"):
            return float("inf")
        return self.dist(node1, node2)

    def plan(self, start: Point3d, goal: Point3d):
        '''
        Interface for planning.
        '''
        raise NotImplementedError
