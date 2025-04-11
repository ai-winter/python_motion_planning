"""
@file: theta_star.py
@breif: Theta* motion planning
@author: Yang Haodong, Wu Maojia
@update: 2024.6.23
"""
import heapq

from .a_star import AStar
from python_motion_planning.utils import Env, Node, Grid


class ThetaStar(AStar):
    """
    Class for Theta* motion planning.

    Parameters:
        start (tuple):
            start point coordinate
        goal (tuple):
            goal point coordinate
        env (Grid):
            environment
        heuristic_type (str):
            heuristic function type

    Examples:
        >>> import python_motion_planning as pmp
        >>> planner = pmp.ThetaStar((5, 5), (45, 25), pmp.Grid(51, 31))
        >>> cost, path, expand = planner.plan()
        >>> planner.plot.animation(path, str(planner), cost, expand)  # animation
        >>> planner.run()       # run both planning and animation

    References:
        [1] Theta*: Any-Angle Path Planning on Grids
        [2] Any-angle path planning on non-uniform costmaps
    """
    def __init__(self, start: tuple, goal: tuple, env: Grid, heuristic_type: str = "euclidean") -> None:
        super().__init__(start, goal, env, heuristic_type)

    def __str__(self) -> str:
        return "Theta*"

    def plan(self) -> tuple:
        """
        Theta* motion plan function.

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

                node_p = CLOSED.get(node.parent)

                if node_p:
                    self.updateVertex(node_p, node_n)

                # goal found
                if node_n == self.goal:
                    heapq.heappush(OPEN, node_n)
                    break
                
                # update OPEN list
                heapq.heappush(OPEN, node_n)

            CLOSED[node.current] = node
        return [], [], []

    def updateVertex(self, node_p: Node, node_c: Node) -> None:
        """
        Update extend node information with current node's parent node.

        Parameters:
            node_p (Node): parent node
            node_c (Node): current node
        """
        if self.lineOfSight(node_c, node_p):
            # path 2
            if node_p.g + self.dist(node_c, node_p) <= node_c.g:
                node_c.g = node_p.g + self.dist(node_c, node_p)
                node_c.parent = node_p.current

    def lineOfSight(self, node1: Node, node2: Node) -> bool:
        """
        Judge collision when moving from node1 to node2 using Bresenham.

        Parameters:
            node1 (Node): start node
            node2 (Node): end node

        Returns:
            line_of_sight (bool): True if line of sight exists ( no collision ) else False
        """
        if node1.current in self.obstacles or node2.current in self.obstacles:
            return False
        
        x1, y1 = node1.current
        x2, y2 = node2.current

        if x1 < 0 or x1 >= self.env.x_range or y1 < 0 or y1 >= self.env.y_range:
            return False
        if x2 < 0 or x2 >= self.env.x_range or y2 < 0 or y2 >= self.env.y_range:
            return False

        d_x = abs(x2 - x1)
        d_y = abs(y2 - y1)
        s_x = 0 if (x2 - x1) == 0 else (x2 - x1) / d_x
        s_y = 0 if (y2 - y1) == 0 else (y2 - y1) / d_y
        x, y, e = x1, y1, 0

        # check if any obstacle exists between node1 and node2
        if d_x > d_y:
            tau = (d_y - d_x) / 2
            while not x == x2:
                if e > tau:
                    x = x + s_x
                    e = e - d_y
                elif e < tau:
                    y = y + s_y
                    e = e + d_x
                else:
                    x = x + s_x
                    y = y + s_y
                    e = e + d_x - d_y
                if (x, y) in self.obstacles:
                    return False
        # swap x and y
        else:
            tau = (d_x - d_y) / 2
            while not y == y2:
                if e > tau:
                    y = y + s_y
                    e = e - d_x
                elif e < tau:
                    x = x + s_x
                    e = e + d_y
                else:
                    x = x + s_x
                    y = y + s_y
                    e = e + d_y - d_x
                if (x, y) in self.obstacles:
                    return False
        
        return True
