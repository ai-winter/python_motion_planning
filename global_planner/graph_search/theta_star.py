'''
@file: theta_star.py
@breif: Theta* motion planning
@author: Winter
@update: 2024.2.2
'''
import os, sys
import heapq

sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))

from .a_star import AStar
from utils import Env, Node

class ThetaStar(AStar):
    '''
    Class for Theta* motion planning.

    Parameters
    ----------
    start: tuple
        start point coordinate
    goal: tuple
        goal point coordinate
    env: Env
        environment
    heuristic_type: str
        heuristic function type, default is euclidean

    Examples
    ----------
    >>> from utils import Grid
    >>> from graph_search import ThetaStar
    >>> start = (5, 5)
    >>> goal = (45, 25)
    >>> env = Grid(51, 31)
    >>> planner = ThetaStar(start, goal, env)
    >>> planner.run()
    '''
    def __init__(self, start: tuple, goal: tuple, env: Env, heuristic_type: str = "euclidean") -> None:
        super().__init__(start, goal, env, heuristic_type)

    def __str__(self) -> str:
        return "Theta*"

    def plan(self):
        '''
        Theta* motion plan function.
        [1] Theta*: Any-Angle Path Planning on Grids
        [2] Any-angle path planning on non-uniform costmaps

        Return
        ----------
        cost: float
            path cost
        path: list
            planning path
        expand: list
            all nodes that planner has searched
        '''
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
                return self.extractPath(CLOSED), CLOSED

            for node_n in self.getNeighbor(node):                
                # exists in CLOSED set
                if node_n in CLOSED:
                    continue
                
                # path1
                node_n.parent = node.current
                node_n.h = self.h(node_n, self.goal)

                try:
                    p_index = CLOSED.index(Node(node.parent))
                    node_p = CLOSED[p_index]
                except:
                    node_p = None

                if node_p:
                    self.updateVertex(node_p, node_n)

                # goal found
                if node_n == self.goal:
                    heapq.heappush(OPEN, node_n)
                    break
                
                # update OPEN set
                heapq.heappush(OPEN, node_n)
            
            CLOSED.append(node)
        return ([], []), []


    def updateVertex(self, node_p: Node, node_c: Node) -> None:
        '''
        Update extend node information with current node's parent node.

        Parameters
        ----------
        node_p, node_c: Node
        '''
        if self.lineOfSight(node_c, node_p):
            # path 2
            if node_p.g + self.dist(node_c, node_p) <= node_c.g:
                node_c.g = node_p.g + self.dist(node_c, node_p)
                node_c.parent = node_p.current
            

    def lineOfSight(self, node1: Node, node2: Node) -> bool:
        '''
        Judge collision when moving from node1 to node2 using Bresenham.

        Parameters
        ----------
        node1, node2: Node

        Return
        ----------
        collision: bool
            True if line of sight exists ( no collision ) else False
        '''
        if node1.current in self.obstacles or node2.current in self.obstacles:
            return False
        
        x1, y1 = node1.current
        x2, y2 = node2.current

        d_x = abs(x2 - x1)
        d_y = abs(y2 - y1)
        s_x = 0 if (x2 - x1) == 0 else (x2 - x1) / d_x
        s_y = 0 if (y2 - y1) == 0 else (y2 - y1) / d_y
        x, y, e = x1, y1, 0

        # check if any obstacle exists between node1 and node2
        if d_x > d_y:
            tao = (d_y - d_x) / 2
            while not x == x2:
                if e > tao:
                    x = x + s_x
                    e = e - d_y
                elif e < tao:
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
            tao = (d_x - d_y) / 2
            while not y == y2:
                if e > tao:
                    y = y + s_y
                    e = e - d_x
                elif e < tao:
                    x = x + s_x
                    e = e + d_y
                else:
                    x = x + s_x
                    y = y + s_y
                    e = e + d_y - d_x
                if (x, y) in self.obstacles:
                    return False
        
        return True