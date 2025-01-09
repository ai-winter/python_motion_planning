"""
@file: bi_a_star.py
@breif: Bidirectional A* motion planning
@author: Yang Haodong
@update: 2024.6.10
"""
import heapq

from .a_star import AStar

from python_motion_planning.common.utils import LOG
from python_motion_planning.common.structure import Node
from python_motion_planning.common.geometry import Point3d

class BiAStar(AStar):
    """
    Class for Bidirectional A* motion planning.

    Parameters:
        params (dict): parameters
    """
    def __init__(self, params: dict) -> None:
        super().__init__(params)
    
    def __str__(self) -> str:
        return "Bidirectional A*"
    
    def plan(self, start: Point3d, goal: Point3d):
        """
        Bidirectional A* motion plan function.

        Returns:
            cost (float): path cost
            path (list): planning path
            expand (list): all nodes that planner has searched
        """
        self.start = Node(start, start, 0, 0)
        self.goal = Node(goal, goal, 0, 0)

        # OPEN set with priority and CLOSED set
        OPEN_F, OPEN_B = [], []
        CLOSED_F, CLOSED_B = dict(), dict()
        heapq.heappush(OPEN_F, self.start)
        heapq.heappush(OPEN_B, self.goal)

        while OPEN_F and OPEN_B:
            node_f = heapq.heappop(OPEN_F)
            while node_f.current in CLOSED_F:
                node_f = heapq.heappop(OPEN_F)

            node_b = heapq.heappop(OPEN_B)
            while node_b.current in CLOSED_B:
                node_b = heapq.heappop(OPEN_B)

            # goal found
            if not self.collision_checker(node_f.current, node_b.current):
                CLOSED_F[node_f.current] = node_f
                CLOSED_B[node_b.current] = node_b
                cost, path, expand = self.extractPath(CLOSED_F, CLOSED_B, node_f, node_b)
                LOG.INFO(f"{str(self)} Planner Planning Successfully. Cost: {cost}")
                return path, [
                    {"type": "value", "data": True, "name": "success"},
                    {"type": "value", "data": cost, "name": "cost"},
                    {"type": "path", "data": path, "name": "normal"},
                    {"type": "grids", "data": [n for n in expand], "name": "expand"}
                ]

            CLOSED_F[node_f.current] = node_f
            CLOSED_B[node_b.current] = node_b

            # forward
            for node_f_n in self.getNeighbor(node_f):                
                # exists in CLOSED set
                if node_f_n.current in CLOSED_F:
                    continue
                
                node_f_n.parent = node_f.current
                node_f_n.h = self.h(node_f_n, node_b)
                
                # update OPEN set
                heapq.heappush(OPEN_F, node_f_n)
            

            # backward
            for node_b_n in self.getNeighbor(node_b):                
                # exists in CLOSED set
                if node_b_n.current in CLOSED_B:
                    continue
                
                node_b_n.parent = node_b.current
                node_b_n.h = self.h(node_b_n, node_f)
                
                # update OPEN set
                heapq.heappush(OPEN_B, node_b_n)
            
        LOG.INFO("Planning Failed.")
        return path, [
            {"type": "value", "data": False, "name": "success"},
            {"type": "value", "data": 0.0, "name": "cost"},
            {"type": "path", "data": [], "name": "normal"},
            {"type": "grids", "data": [], "name": "expand"}
        ]
    
    def extractPath(self, closed_set_f: dict, closed_set_b: dict, node_f: Node, node_b: Node):
        """
        Extract the path based on the CLOSED set.

        Parameters:
            closed_set_f (dict): FORWARD CLOSED set
            closed_set_b (dict): BACKWARD CLOSED set
            node_f (node): backtracking node in forward-searching
            node_b (node): backtracking node in backward-searching

        Returns:
            cost (float): the cost of planning path
            path (list): the planning path
        """
        cost = 0

        expand = []
        tree_size = max(len(closed_set_f), len(closed_set_b))
        for k in range(tree_size):
            if k < len(closed_set_f):
                expand.append(list(closed_set_f.items())[k][1].current)
            if k < len(closed_set_b):
                expand.append(list(closed_set_b.items())[k][1].current)

        # forward
        node = closed_set_f[node_f.current]
        path_f = [node.current]
        while node != self.start:
            node_parent = closed_set_f[node.parent]
            cost += self.dist(node, node_parent)
            node = node_parent
            path_f.append(node.current)
        
        # backward
        node = closed_set_b[node_b.current]
        path_b = []
        while node != self.goal:
            node_parent = closed_set_b[node.parent]
            cost += self.dist(node, node_parent)
            node = node_parent
            path_b.append(node.current)

        return cost, list(reversed(path_f)) + path_b, expand