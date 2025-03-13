"""
@file: gbfs_planner.py
@breif: Greedy Best First Search path planning
@author: Yang Haodong
@update: 2024.2.11
"""
import heapq

from typing import List, Tuple, Dict

from .astar_planner import AStarPlanner

from python_motion_planning.common.utils import LOG
from python_motion_planning.common.structure import Node, Env
from python_motion_planning.common.geometry import Point3d

class GBFSPlanner(AStarPlanner):
    """
    Class for GBFS motion planning.

    Parameters:
        env (Env): environment object
        params (dict): parameters
    """
    def __init__(self, env: Env, params: dict) -> None:
        super().__init__(env, params)
    
    def __str__(self) -> str:
        return "Greedy Best First Search(GBFS)"

    def plan(self, start: Point3d, goal: Point3d) -> Tuple[List[Point3d], List[Dict]]:
        """
        Greedy Best First Search motion plan function.

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
             
                # hit the obstacle
                if node_n.current in self.obstacles:
                    continue
                
                # exists in CLOSED set
                if node_n.current in CLOSED:
                    continue
                
                node_n.parent = node.current
                node_n.h = self.h(node_n, self.goal)
                node_n.g = 0

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