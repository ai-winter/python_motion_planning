'''
@file: theta_star.py
@breif: Theta* motion planning
@author: Winter
@update: 2023.8.25
'''
import os, sys
import heapq

sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))

from .graph_search import GraphSearcher
from utils import Env, Node

class ThetaStar(GraphSearcher):
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


    def updateVertex(self, node_p, node_c):
        if not self.lineOfSight(node_c, node_p):
            # path 2
            if node_p.g + self.dist(node_c, node_p) < node_c.g:
                node_c.g = node_p.g + self.dist(node_c, node_p)
                node_c.parent = node_p.current
            

    def lineOfSight(self, node1, node2):
        '''
        Judge collision when moving from node1 to node2 using Bresenham.

        Parameters
        ----------
        node1, node2: Node

        Return
        ----------
        collision: bool
            True if collision exists else False
        '''
        if node1.current in self.obstacles or node2.current in self.obstacles:
            return True
        
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
                    return True
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
                    return True
        
        return False

    def getNeighbor(self, node: Node) -> list:
        '''
        Find neighbors of node.

        Parameters
        ----------
        node: Node
            current node

        Return
        ----------
        neighbors: list
            neighbors of current node
        '''
        return [node + motion for motion in self.motions
                if not self.isCollision(node, node + motion)]

    def extractPath(self, closed_set):
        '''
        Extract the path based on the CLOSED set.

        Parameters
        ----------
        closed_set: list
            CLOSED set

        Return
        ----------
        cost: float
            the cost of planning path
        path: list
            the planning path
        '''
        cost = 0
        node = closed_set[closed_set.index(self.goal)]
        path = [node.current]
        while node != self.start:
            node_parent = closed_set[closed_set.index(Node(node.parent, None, None, None))]
            cost += self.dist(node, node_parent)
            node = node_parent
            path.append(node.current)
        return cost, path

    def run(self):
        '''
        Running both plannig and animation.
        '''
        (cost, path), expand = self.plan()
        self.plot.animation(path, str(self), cost, expand)
