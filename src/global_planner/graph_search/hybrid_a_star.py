'''
@file: hybrid_a_star.py
@breif: Hybrid A* motion planning
@author: Winter
@update: 2024.1.4
'''
import math
import os, sys

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../")))

from .graph_search import GraphSearcher
from src.utils import Env, Node

class HybridAStar(GraphSearcher):
    '''
    Class for Hybrid A* motion planning.

    Parameters
    ----------
    start: tuple
        start point coordinate
    goal: tuple
        goal point coordinate
    env: Env
        environment

    Examples
    ----------
    >>> from src.utils import Grid
    >>> from graph_search import HybridAStar
    >>> start = (5, 5)
    >>> goal = (45, 25)
    >>> env = Grid(51, 31)
    >>> planner = HybridAStar(start, goal, env)
    >>> planner.run()
    '''
    def __init__(self, start: tuple, goal: tuple, env: Env, penalty_turn: float,
        penalty_cod: float, penalty_reverse: float, headings: int) -> None:
        super().__init__(start, goal, env, "euclidean")
        self.penalty_turn = penalty_turn
        self.penalty_cod = penalty_cod
        self.penalty_reverse = penalty_reverse
        self.headings = headings
        self.delta_heading = 2 * math.pi / headings

    def __str__(self) -> str:
        return "A*"

    class HybridNode(Node):
        '''
        Class for Hybrid A* nodes.

        Parameters
        ----------
        current: tuple
            current coordinate
        parent: tuple
            coordinate of parent node
        g: float
            path cost
        h: float
            heuristic cost
        reverse: bool
            whether reverse is allowed or not
        '''
        def __init__(self, current: tuple, parent: tuple=None, g: float=0, h: float=0,
            reverse: bool=False) -> None:
            assert len(current) == 3
            self.reverse = reverse
            super().__init__(current, parent, g, h)
            self.penalty_turn = None
            self.penalty_cod = None
            self.penalty_reverse = None

        def propsInit(self, penalty_turn: float, penalty_cod: float, penalty_reverse: float):
            self.penalty_turn = penalty_turn
            self.penalty_cod = penalty_cod
            self.penalty_reverse = penalty_reverse

        def __add__(self, node):
            x = self.x + node.x * math.cos(self.theta) - node.y * math.sin(self.theta)
            y = self.y + node.x * math.sin(self.theta) + node.y * math.cos(self.theta)
            t = self.theta + node.theta

            g = node.x
            if self.reverse:
                g *= self.penalty_reverse
            if self.reverse != node.reverse:
                g *= self.penalty_cod
            if self.parent is not None and t != self.ptheta:
                g *= self.penalty_turn

            return Node((x, y, t), self.current, self.g + g, self.h)

        @property
        def theta(self) -> float:
            return self.current[2]
        
        @property
        def ptheta(self) -> float:
            return self.parrent[2]

    def plan(self):
        '''
        Hybrid A* motion plan function.
        [1] Practical Search Techniques in Path Planning for Autonomous Driving

        Return
        ----------
        cost: float
            path cost
        path: list
            planning path
        expand: list
            all nodes that planner has searched
        '''
        pass

    
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
        dy = [0,        -0.0415893,  0.0415893]
        dx = [0.7068582,   0.705224,   0.705224]
        dt = [0,         0.1178097,   -0.1178097]
        motions = [
            self.HybridNode((dx[0], dy[0], dt[0]), reverse=False),
            self.HybridNode((dx[1], dy[1], dt[1]), reverse=False),
            self.HybridNode((dx[2], dy[2], dt[2]), reverse=False),
            self.HybridNode((-dx[0], dy[0], -dt[0]), reverse=True),
            self.HybridNode((-dx[1], dy[1], -dt[1]), reverse=True),
            self.HybridNode((-dx[2], dy[2], -dt[2]), reverse=True),
        ]


        return [node + motion for motion in self.motions
                if not self.isCollision(node, node + motion)]


    def updateG(self, node: HybridNode):
        pass

    def updateH(self, node: HybridNode):
        pass