'''
@file: s_theta_star.py
@breif: S-Theta* motion planning
@author: Wu Maojia
@update: 2024.6.23
'''
import heapq
from math import acos

from .theta_star import ThetaStar
from python_motion_planning.utils import Env, Node, Grid


class SThetaStar(ThetaStar):
    """
    Class for S-Theta* motion planning.

    Parameters:
        start (tuple): start point coordinate
        goal (tuple): goal point coordinate
        env (Grid): environment
        heuristic_type (str): heuristic function type

    Examples:
        >>> import python_motion_planning as pmp
        >>> planner = pmp.SThetaStar((5, 5), (45, 25), pmp.Grid(51, 31))
        >>> cost, path, expand = planner.plan()
        >>> planner.plot.animation(path, str(planner), cost, expand)  # animation
        >>> planner.run()       # run both planning and animation

    References:
        [1] S-Theta*: low steering path-planning algorithm
    """

    def __init__(self, start: tuple, goal: tuple, env: Grid, heuristic_type: str = "euclidean") -> None:
        super().__init__(start, goal, env, heuristic_type)

    def __str__(self) -> str:
        return "S-Theta*"

    def plan(self) -> tuple:
        """
        S-Theta* motion plan function.

        Returns:
            cost (float): path cost
            path (list): planning path
            expand (list): all nodes that planner has searched
        """
        # OPEN list (priority queue) and CLOSED list (hash table)
        OPEN = []
        heapq.heappush(OPEN, self.start)
        CLOSED = dict()

        while OPEN:
            node = heapq.heappop(OPEN)

            # exists in CLOSED list
            if node.current in CLOSED:
                continue

            # goal found
            if node == self.goal:
                CLOSED[node.current] = node
                cost, path = self.extractPath(CLOSED)
                return cost, path, list(CLOSED.values())

            for node_n in self.getNeighbor(node):
                # exists in CLOSED list
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

                if node_p:
                    self.updateVertex(node_p, node_n, alpha)

                # goal found
                if node_n == self.goal:
                    heapq.heappush(OPEN, node_n)
                    break

                # update OPEN list
                heapq.heappush(OPEN, node_n)

            CLOSED[node.current] = node
        return [], [], []

    def updateVertex(self, node_p: Node, node_c: Node, alpha: float) -> None:
        """
        Update extend node information with current node's parent node.

        Parameters:
            node_p (Node): parent node
            node_c (Node): current node
            alpha (float): alpha angle
        """
        # if alpha == 0 or self.lineOfSight(node_c, node_p):    # "alpha == 0" will cause the path to penetrate obstacles
        if self.lineOfSight(node_c, node_p):
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