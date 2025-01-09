"""
@file: lazy_theta_star.py
@breif: Lazy Theta* motion planning
@author: Yang Haodong, Wu Maojia
@update: 2024.2.11
"""
import heapq

from .theta_star import ThetaStar

from python_motion_planning.common.utils import LOG
from python_motion_planning.common.structure import Node
from python_motion_planning.common.geometry import Point3d

class LazyThetaStar(ThetaStar):
    """
    Class for Lazy Theta* motion planning.

    Parameters:
        params (dict): parameters

    References:
        [1] Lazy Theta*: Any-Angle Path Planning and Path Length Analysis in 3D
    """
    def __init__(self, params: dict) -> None:
        super().__init__(params)

    def __str__(self) -> str:
        return "Lazy Theta*"

    def plan(self, start: Point3d, goal: Point3d):
        """
        Lazy Theta* motion plan function.

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

            # set vertex: path 1
            try:
                node_p = CLOSED.get(node.parent)
                if node_p and self.collision_checker(node_p.current, node.current):
                    node.g = float("inf")
                    for node_n in self.getNeighbor(node):
                        if node_n.current in CLOSED:
                            node_n = CLOSED[node_n.current]
                            if node.g > node_n.g + self.dist(node_n, node):
                                node.g = node_n.g + self.dist(node_n, node)
                                node.parent = node_n.current
            except:
                pass

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
                
                # path1
                node_n.parent = node.current
                node_n.h = self.h(node_n, self.goal)
                node_p = CLOSED.get(node.parent)

                if node_p:
                    # path2
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
        # path 2
        if node_p.g + self.dist(node_c, node_p) <= node_c.g:
            node_c.g = node_p.g + self.dist(node_c, node_p)
            node_c.parent = node_p.current  
