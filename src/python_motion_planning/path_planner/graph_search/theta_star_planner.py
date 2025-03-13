"""
@file: theta_star_planner.py
@breif: Theta* path planning
@author: Yang Haodong, Wu Maojia
@update: 2024.2.11
"""
import heapq

from typing import List, Tuple, Dict

from .astar_planner import AStarPlanner

from python_motion_planning.common.utils import LOG
from python_motion_planning.common.structure import Node, Env
from python_motion_planning.common.geometry import Point3d

class ThetaStarPlanner(AStarPlanner):
    """
    Class for Theta* path planning.

    Parameters:
        env (Env): environment object
        params (dict): parameters

    References:
        [1] Theta*: Any-Angle Path Planning on Grids
        [2] Any-angle path planning on non-uniform costmaps
    """
    def __init__(self, env: Env, params: dict) -> None:
        super().__init__(env, params)

    def __str__(self) -> str:
        return "Theta*"

    def plan(self, start: Point3d, goal: Point3d) -> Tuple[List[Point3d], List[Dict]]:
        """
        Theta* motion plan function.

        Parameters:
            start (Point3d): The starting point of the planning path.
            goal (Point3d): The goal point of the planning path.

        Returns:
            path (List[Point3d]): The planned path from start to goal.
            visual_info (List[Dict]): Information for visualization
        """
        self.start = Node(start, start, 0, 0)
        self.goal = Node(goal, goal, 0, 0)

        # OPEN set with priority and CLOSED set
        OPEN = []
        heapq.heappush(OPEN, self.start)
        CLOSED = dict()

        while OPEN:
            node = heapq.heappop(OPEN)

            # exists in CLOSED set
            if node.current in CLOSED:
                continue

            # goal found
            if node == self.goal:
                CLOSED[self.goal.current] = node
                cost, path = self.extractPath(CLOSED)
                LOG.INFO(f"{str(self)} PathPlanner Planning Successfully. Cost: {cost}")
                return path, [
                    {"type": "value", "data": True, "name": "success"},
                    {"type": "value", "data": cost, "name": "cost"},
                    {"type": "path", "data": path, "name": "normal"},
                    {"type": "grids", "data": [n.current for n in CLOSED.values()], "name": "expand"}
                ]

            for node_n in self.getNeighbor(node):                
                # exists in CLOSED set
                if node_n.current in CLOSED:
                    continue
                
                # path1
                node_n.parent = node.current
                node_n.h = self.h(node_n, self.goal)
                node_p = CLOSED.get(node.parent)

                if node_p:
                    self.updateVertex(node_p, node_n)

                # goal found
                if node_n == self.goal:
                    heapq.heappush(OPEN, node_n)
                    break
                
                # update OPEN set
                heapq.heappush(OPEN, node_n)
            
            CLOSED[node.current] = node

        LOG.INFO("Planning Failed.")
        return path, [
            {"type": "value", "data": False, "name": "success"},
            {"type": "value", "data": 0.0, "name": "cost"},
            {"type": "path", "data": [], "name": "normal"},
            {"type": "grids", "data": [], "name": "expand"}
        ]

    def updateVertex(self, node_p: Node, node_c: Node) -> None:
        """
        Update extend node information with current node's parent node.

        Parameters:
            node_p (Node): parent node
            node_c (Node): current node
        """
        if not self.collision_checker(node_c.current, node_p.current):
            # path 2
            if node_p.g + self.dist(node_c, node_p) <= node_c.g:
                node_c.g = node_p.g + self.dist(node_c, node_p)
                node_c.parent = node_p.current