'''
@file: s_theta_star.py
@breif: S-Theta* motion planning
@author: Wu Maojia
@update: 2024.3.6
'''
import heapq
from math import acos

from .theta_star import ThetaStar

from python_motion_planning.common.utils import LOG
from python_motion_planning.common.structure import Node
from python_motion_planning.common.geometry import Point3d

class SThetaStar(ThetaStar):
    """
    Class for S-Theta* motion planning.

    Parameters:
        params (dict): parameters

    References:
        [1] S-Theta*: low steering path-planning algorithm
    """

    def __init__(self, params: dict) -> None:
        super().__init__(params)

    def __str__(self) -> str:
        return "S-Theta*"

    def plan(self, start: Point3d, goal: Point3d):
        """
        S-Theta* motion plan function.

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

                # path1
                node_n.parent = node.current
                node_n.h = self.h(node_n, self.goal)

                alpha = 0.0
                node_p = CLOSED.get(node.parent)

                if node_p:
                    alpha = self.getAlpha(node_p, node_n)
                    node_n.g += alpha
                    self.updateVertex(node_p, node_n, alpha)

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

    def updateVertex(self, node_p: Node, node_c: Node, alpha: float) -> None:
        """
        Update extend node information with current node's parent node.

        Parameters:
            node_p (Node): parent node
            node_c (Node): current node
            alpha (float): alpha angle
        """
        if not self.collision_checker(node_c.current, node_p.current):
            # path 2
            new_g = node_p.g + self.dist(node_c, node_p) + alpha
            if new_g <= node_c.g:
                node_c.g = new_g
                node_c.parent = node_p.current

    def getAlpha(self, node_p: Node, node_c: Node):
        """
        α(t) represents the deviation in the trajectory to reach the goal node g
        through the node t in relation to the straight-line distance between the parent of its
        predecessor (t ∈ succ(p) and parent(p) = q) and the goal node.

        Parameters:
            node_p (Node): parent node
            node_c (Node): current node

        Returns:
            alpha (float): alpha angle
        """
        d_qt = self.dist(node_p, node_c)
        d_qg = self.dist(node_p, self.goal)
        d_tg = self.dist(node_c, self.goal)
        value = (d_qt * d_qt + d_qg * d_qg - d_tg * d_tg) / (2.0 * d_qt * d_qg)
        value = max(-1.0, min(1.0, value))
        cost = acos(value)
        return cost