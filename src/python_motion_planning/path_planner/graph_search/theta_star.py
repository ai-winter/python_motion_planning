"""
@file: theta_star.py
@breif: Theta* motion planning
@author: Yang Haodong, Wu Maojia
@update: 2024.6.23
"""
from typing import Union
import heapq

from python_motion_planning.common import BaseMap, Grid, Node, TYPES
from .a_star import AStar


class ThetaStar(AStar):
    """
    Class for Theta* path planner.

    Args:
        *args: see the parent class.
        **kwargs: see the parent class.

    References:
        [1] Theta*: Any-Angle Path Planning on Grids
        [2] Any-angle path planning on non-uniform costmaps
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __str__(self) -> str:
        return "Theta*"

    def plan(self) -> tuple:
        """
        Interface for planning.

        Returns:
            path: A list containing the path waypoints
            path_info: A dictionary containing the path information (success, length, cost, expand)
        """
        # OPEN list (priority queue) and CLOSED list (hash table)
        OPEN = []
        start_node = Node(self.start, None, 0, self.get_heuristic(self.start))
        heapq.heappush(OPEN, start_node)
        CLOSED = dict()

        while OPEN:
            node = heapq.heappop(OPEN)

            # obstacle found
            if not self.map_.is_expandable(node.current, node.parent):
                continue

            # exists in CLOSED list
            if node.current in CLOSED:
                continue

            # goal found
            if node.current == self.goal:
                CLOSED[node.current] = node
                path, length, cost = self.extract_path(CLOSED)
                return path, {
                    "success": True, 
                    "start": self.start, 
                    "goal": self.goal, 
                    "length": length, 
                    "cost": cost, 
                    "expand": CLOSED
                }

            for node_n in self.map_.get_neighbors(node): 
                # exists in CLOSED list
                if node_n.current in CLOSED:
                    continue

                # path1: normal A*
                node_n.g = node.g + self.get_cost(node.current, node_n.current)
                node_n.h = self.get_heuristic(node_n.current)

            
                # path 2: Theta* line of sight update
                node_p = CLOSED.get(node.parent)
                if node_p:
                    self.updateVertex(node_p, node_n)

                # goal found
                if node_n.current == self.goal:
                    heapq.heappush(OPEN, node_n)
                    break
                
                # update OPEN list
                heapq.heappush(OPEN, node_n)

            CLOSED[node.current] = node

        self.failed_info[1]["expand"] = CLOSED
        return self.failed_info

    def updateVertex(self, node_p: Node, node_n: Node) -> None:
        """
        Update extend node information with current node's parent node.

        Args:
            node_p (Node): parent node
            node_n (Node): next node
        """
        if not self.map_.in_collision(node_p.current, node_n.current):
            if node_p.g + self.get_cost(node_p.current, node_n.current) <= node_n.g:
                node_n.g = node_p.g + self.get_cost(node_p.current, node_n.current)
                node_n.parent = node_p.current
