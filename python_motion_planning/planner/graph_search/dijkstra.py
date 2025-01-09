"""
@file: dijkstra.py
@breif: Dijkstra motion planning
@author: Yang Haodong, Wu Maojia
@update: 2024.2.11
"""
import heapq

from .a_star import AStar

from python_motion_planning.common.utils import LOG
from python_motion_planning.common.structure import Node
from python_motion_planning.common.geometry import Point3d

class Dijkstra(AStar):
    """
    Class for Dijkstra motion planning.

    Parameters:
        params (dict): parameters

    References:
        [1] A Formal Basis for the heuristic Determination of Minimum Cost Paths
    """
    def __init__(self, params: dict) -> None:
        super().__init__(params)
    
    def __str__(self) -> str:
        return "Dijkstra"

    def plan(self, start: Point3d, goal: Point3d):
        """
        Class for Dijkstra motion planning.

        Parameters:
            start (tuple): start point coordinate
            goal (tuple): goal point coordinate
            env (Env): environment
            heuristic_type (str): heuristic function type

        Examples:
            >>> from python_motion_planning.utils import Grid
            >>> from graph_search import Dijkstra
            >>> start = (5, 5)
            >>> goal = (45, 25)
            >>> env = Grid(51, 31)
            >>> planner = Dijkstra(start, goal, env)
            >>> planner.run()
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
             
                # hit the obstacle
                if node_n.current in self.obstacles:
                    continue
                
                # exists in CLOSED set
                if node_n.current in CLOSED:
                    continue
                
                node_n.parent = node.current
                node_n.h = 0

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