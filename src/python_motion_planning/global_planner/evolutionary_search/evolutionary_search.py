"""
@file: evolutionary_search.py
@breif: Base class for planner based on evolutionary searching
@author: Winter
@update: 2023.7.13
"""
import math

from python_motion_planning.utils import Env, Node, Planner, Grid

class EvolutionarySearcher(Planner):
    """
    Base class for planner based on evolutionary searching.

    Parameters:
        start (tuple): start point coordinate
        goal (tuple): goal point coordinate
        env (Grid): environment
        heuristic_type (str): heuristic function type
    """
    def __init__(self, start: tuple, goal: tuple, env: Grid, heuristic_type: str="euclidean") -> None:
        super().__init__(start, goal, env)
        # heuristic type
        self.heuristic_type = heuristic_type
        # allowed motions
        self.motions = self.env.motions
        # obstacles
        self.obstacles = self.env.obstacles

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

    def cost(self, node1: Node, node2: Node) -> float:
        """
        Calculate cost for this motion.

        Parameters:
            node1 (Node): node 1
            node2 (Node): node 2

        Returns:
            cost (float): cost of this motion
        """
        if self.isCollision(node1, node2):
            return float("inf")
        return self.dist(node1, node2)

    def isCollision(self, node1: Node, node2: Node) -> bool:
        """
        Judge collision when moving from node1 to node2.

        Parameters:
            node1 (Node): node 1
            node2 (Node): node 2

        Returns:
            collision (bool): True if collision exists else False
        """
        if node1.current in self.obstacles or node2.current in self.obstacles:
            return True

        x1, y1 = node1.current
        x2, y2 = node2.current

        if x1 != x2 and y1 != y2:
            if x2 - x1 == y1 - y2:
                s1 = (min(x1, x2), min(y1, y2))
                s2 = (max(x1, x2), max(y1, y2))
            else:
                s1 = (min(x1, x2), max(y1, y2))
                s2 = (max(x1, x2), min(y1, y2))
            if s1 in self.obstacles or s2 in self.obstacles:
                return True
        return False
