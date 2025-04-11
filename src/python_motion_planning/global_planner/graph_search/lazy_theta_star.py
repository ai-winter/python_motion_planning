"""
@file: lazy_theta_star.py
@breif: Lazy Theta* motion planning
@author: Yang Haodong, Wu Maojia
@update: 2024.6.23
"""
import heapq

from .theta_star import ThetaStar
from python_motion_planning.utils import Env, Node, Grid

class LazyThetaStar(ThetaStar):
    """
    Class for Lazy Theta* motion planning.

    Parameters:
        start (tuple): start point coordinate
        goal (tuple): goal point coordinate
        env (Grid): environment
        heuristic_type (str): heuristic function type

    Examples:
        >>> import python_motion_planning as pmp
        >>> planner = pmp.LazyThetaStar((5, 5), (45, 25), pmp.Grid(51, 31))
        >>> cost, path, expand = planner.plan()
        >>> planner.plot.animation(path, str(planner), cost, expand)  # animation
        >>> planner.run()       # run both planning and animation

    References:
        [1] Lazy Theta*: Any-Angle Path Planning and Path Length Analysis in 3D
    """
    def __init__(self, start: tuple, goal: tuple, env: Grid, heuristic_type: str = "euclidean") -> None:
        super().__init__(start, goal, env, heuristic_type)

    def __str__(self) -> str:
        return "Lazy Theta*"

    def plan(self) -> tuple:
        """
        Lazy Theta* motion plan function.

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

            # set vertex: path 1
            node_p = CLOSED.get(node.parent)
            if node_p:
                if not self.lineOfSight(node_p, node):
                    node.g = float("inf")
                    for node_n in self.getNeighbor(node):
                        if node_n.current in CLOSED:
                            node_n = CLOSED[node_n.current]
                            if node.g > node_n.g + self.dist(node_n, node):
                                node.g = node_n.g + self.dist(node_n, node)
                                node.parent = node_n.current

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
                    # path2
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
        # path 2
        if node_p.g + self.dist(node_c, node_p) <= node_c.g:
            node_c.g = node_p.g + self.dist(node_c, node_p)
            node_c.parent = node_p.current  
