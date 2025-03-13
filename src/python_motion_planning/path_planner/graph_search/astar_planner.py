"""
@file: astar_planner.py
@breif: A* path planning
@author: Yang Haodong
@update: 2024.2.11
"""
import math
import heapq

from typing import List, Tuple, Dict

from python_motion_planning.path_planner import PathPlanner
from python_motion_planning.common.utils import LOG
from python_motion_planning.common.structure import Node, Env
from python_motion_planning.common.geometry import Point3d

class AStarPlanner(PathPlanner):
    """
    Class for A* motion planning.

    Parameters:
        env (Env): environment object
        params (dict): parameters

    References:
        [1] A Formal Basis for the heuristic Determination of Minimum Cost Paths
    """
    def __init__(self, env: Env, params: dict) -> None:
        super().__init__(env, params)
        # allowed motions
        self.motions = [
            Node(Point3d(-1, 0, 0), None, 1, None), Node(Point3d(-1,  1, 0), None, math.sqrt(2), None),
            Node(Point3d( 0, 1, 0), None, 1, None), Node(Point3d( 1,  1, 0), None, math.sqrt(2), None),
            Node(Point3d( 1, 0, 0), None, 1, None), Node(Point3d( 1, -1, 0), None, math.sqrt(2), None),
            Node(Point3d( 0,-1, 0), None, 1, None), Node(Point3d(-1, -1, 0), None, math.sqrt(2), None)
        ]

    def __str__(self) -> str:
        return "A*"

    def h(self, node: Node, goal: Node) -> float:
        """
        Calculate heuristic.

        Parameters:
            node (Node): current node
            goal (Node): goal node

        Returns:
            h (float): heuristic function value of node
        """
        if self.heuristic_type == "manhattan":
            return abs(goal.x - node.x) + abs(goal.y - node.y)
        elif self.heuristic_type == "euclidean":
            return math.hypot(goal.x - node.x, goal.y - node.y)

    def plan(self, start: Point3d, goal: Point3d) -> Tuple[List[Point3d], List[Dict]]:
        """
        A* motion plan function.

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
                
                node_n.parent = node.current
                node_n.h = self.h(node_n, self.goal)

                # goal found
                if node_n == self.goal:
                    heapq.heappush(OPEN, node_n)
                    break
                
                # update OPEN set
                heapq.heappush(OPEN, node_n)
            
            CLOSED[node.current] = node
        
        LOG.INFO("Planning Failed.")
        return [], [
            {"type": "value", "data": False, "name": "success"},
            {"type": "value", "data": 0.0, "name": "cost"},
            {"type": "path", "data": [], "name": "normal"},
            {"type": "grids", "data": [], "name": "expand"}
        ]

    def getNeighbor(self, node: Node) -> list:
        """
        Find neighbors of node.

        Parameters:
            node (Node): current node

        Returns:
            neighbors (list): neighbors of current node
        """
        return [
            node + motion for motion in self.motions if not
            self.collision_checker(node.current, (node + motion).current, "onestep")
        ]

    def extractPath(self, closed_set: dict):
        """
        Extract the path based on the CLOSED set.

        Parameters:
            closed_set (dict): CLOSED set

        Returns:
            cost (float): the cost of planning path
            path (list): the planning path
        """
        cost = 0
        node = closed_set[self.goal.current]
        path = [node.current]
        while node != self.start:
            node_parent = closed_set[node.parent]
            cost += self.dist(node, node_parent)
            node = node_parent
            path.append(node.current)
        return cost, list(reversed(path))
