"""
@file: a_star.py
@breif: A* motion planning
@author: Yang Haodong
@update: 2024.2.11
"""
import math
import heapq

from python_motion_planning.planner import Planner
from python_motion_planning.common.utils import LOG
from python_motion_planning.common.structure import Node
from python_motion_planning.common.geometry import Point3d

class AStar(Planner):
    """
    Class for A* motion planning.

    Parameters:
        params (dict): parameters

    References:
        [1] A Formal Basis for the heuristic Determination of Minimum Cost Paths
    """
    def __init__(self, params: dict) -> None:
        super().__init__(params)
        # allowed motions
        self.motions = [
            Node(Point3d(-1, 0, 0), None, 1, None), Node(Point3d(-1,  1, 0), None, math.sqrt(2), None),
            Node(Point3d( 0, 1, 0), None, 1, None), Node(Point3d( 1,  1, 0), None, math.sqrt(2), None),
            Node(Point3d( 1, 0, 0), None, 1, None), Node(Point3d( 1, -1, 0), None, math.sqrt(2), None),
            Node(Point3d( 0,-1, 0), None, 1, None), Node(Point3d(-1, -1, 0), None, math.sqrt(2), None)
        ]

        # parameters
        for k, v in self.params["strategy"]["planner"].items():
            setattr(self, k, v)

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

    def plan(self, start: Point3d, goal: Point3d):
        """
        A* motion plan function.

        Returns:
            cost (float): path cost
            path (list): planning path
            expand (list): all nodes that planner has searched
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
                LOG.INFO(f"{str(self)} Planner Planning Successfully. Cost: {cost}")
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
