'''
@file: s_theta_star.py
@breif: S-Theta* motion planning
@author: Wu Maojia
@update: 2024.3.6
'''
import heapq
from math import acos

from .theta_star import ThetaStar
from python_motion_planning.utils import Env, Node


class SThetaStar(ThetaStar):
    """
    Class for S-Theta* motion planning.

    Parameters:
        start (tuple): start point coordinate
        goal (tuple): goal point coordinate
        env (Env): environment
        heuristic_type (str): heuristic function type

    Examples:
        >>> from python_motion_planning.utils import Grid
        >>> from graph_search import SThetaStar
        >>> start = (5, 5)
        >>> goal = (45, 25)
        >>> env = Grid(51, 31)
        >>> planner = SThetaStar(start, goal, env)
        >>> planner.run()

    References:
        [1] S-Theta*: low steering path-planning algorithm
    """

    def __init__(self, start: tuple, goal: tuple, env: Env, heuristic_type: str = "euclidean") -> None:
        super().__init__(start, goal, env, heuristic_type)

    def __str__(self) -> str:
        return "S-Theta*"

    def plan(self):
        """
        S-Theta* motion plan function.

        Returns:
            cost (float): path cost
            path (list): planning path
            expand (list): all nodes that planner has searched
        """
        # OPEN set with priority and CLOSED set
        OPEN = []
        heapq.heappush(OPEN, self.start)
        CLOSED = []

        while OPEN:
            node = heapq.heappop(OPEN)

            # exists in CLOSED set
            if node in CLOSED:
                continue

            # goal found
            if node == self.goal:
                CLOSED.append(node)
                cost, path = self.extractPath(CLOSED)
                return cost, path, CLOSED

            for node_n in self.getNeighbor(node):
                # exists in CLOSED set
                if node_n in CLOSED:
                    continue

                # path1
                node_n.parent = node.current
                node_n.h = self.h(node_n, self.goal)

                alpha = 0.0
                try:
                    p_index = CLOSED.index(Node(node.parent))
                    node_p = CLOSED[p_index]
                    alpha = self.getAlpha(node_p, node_n)
                    node_n.g += alpha
                except:
                    node_p = None

                if node_p:
                    self.updateVertex(node_p, node_n, alpha)

                # goal found
                if node_n == self.goal:
                    heapq.heappush(OPEN, node_n)
                    break

                # update OPEN set
                heapq.heappush(OPEN, node_n)

            CLOSED.append(node)
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