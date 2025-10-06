"""
@file: theta_star.py
@author: Wu Maojia, Yang Haodong
@update: 2025.10.6
"""
from typing import Union, List, Tuple, Dict, Any
import heapq

from python_motion_planning.common import BaseMap, Grid, Node, TYPES
from python_motion_planning.path_planner.graph_search.a_star import AStar


class ThetaStar(AStar):
    """
    Class for Theta* path planner.

    Args:
        *args: see the parent class.
        **kwargs: see the parent class.

    References:
        [1] Theta*: Any-Angle Path Planning on Grids
        [2] Any-angle path planning on non-uniform costmaps

    Examples:
        >>> map_ = Grid(bounds=[[0, 15], [0, 15]])
        >>> planner = ThetaStar(map_=map_, start=(5, 5), goal=(10, 10))
        >>> planner.plan()
        ([(5, 5), (8, 8), (9, 9), (10, 10)], {'success': True, 'start': (5, 5), 'goal': (10, 10), 'length': 7.0710678118654755, 'cost': 7.0710678118654755, 'expand': {(5, 5): Node((5, 5), None, 0, 7.0710678118654755), (6, 6): Node((6, 6), (5, 5), 1.4142135623730951, 5.656854249492381), (7, 7): Node((7, 7), (5, 5), 2.8284271247461903, 4.242640687119285), (8, 8): Node((8, 8), (5, 5), 4.242640687119285, 2.8284271247461903), (9, 9): Node((9, 9), (8, 8), 5.65685424949238, 1.4142135623730951), (10, 10): Node((10, 10), (9, 9), 7.071067811865475, 0.0)}})

        >>> planner.map_.type_map[3:10, 6] = TYPES.OBSTACLE
        >>> planner.plan()
        ([(5, 5), (8, 8), (9, 9), (10, 10)], {'success': True, 'start': (5, 5), 'goal': (10, 10), 'length': 7.0710678118654755, 'cost': 7.0710678118654755, 'expand': {(5, 5): Node((5, 5), None, 0, 7.0710678118654755), (6, 6): Node((6, 6), (5, 5), 1.4142135623730951, 5.656854249492381), (7, 7): Node((7, 7), (5, 5), 2.8284271247461903, 4.242640687119285), (8, 8): Node((8, 8), (5, 5), 4.242640687119285, 2.8284271247461903), (9, 9): Node((9, 9), (8, 8), 5.65685424949238, 1.4142135623730951), (10, 10): Node((10, 10), (9, 9), 7.071067811865475, 0.0)}})
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __str__(self) -> str:
        return "Theta*"

    def plan(self) -> Union[List[Tuple[float, ...]], Dict[str, Any]]:
        """
        Interface for planning.

        Returns:
            path: A list containing the path waypoints
            path_info: A dictionary containing the path information (success, length, cost, expand)
        """
        # OPEN list (priority queue) and CLOSED list (hash table)
        OPEN = []
        start_node = Node(self.start, None, 0, self.get_heuristic(self.start))
        heapq.heappush(OPEN, start_node)
        CLOSED = dict()

        while OPEN:
            node = heapq.heappop(OPEN)

            # exists in CLOSED list
            if node.current in CLOSED:
                continue

            # goal found
            if node.current == self.goal:
                CLOSED[node.current] = node
                path, length, cost = self.extract_path(CLOSED)
                return path, {
                    "success": True, 
                    "start": self.start, 
                    "goal": self.goal, 
                    "length": length, 
                    "cost": cost, 
                    "expand": CLOSED
                }

            for node_n in self.map_.get_neighbors(node, diagonal=self.diagonal): 
                # exists in CLOSED list
                if node_n.current in CLOSED:
                    continue

                # path1: normal A*
                node_n.g = node.g + self.get_cost(node.current, node_n.current)
                node_n.h = self.get_heuristic(node_n.current)

            
                # path 2: Theta* line of sight update
                node_p = CLOSED.get(node.parent)
                if node_p:
                    if not self.map_.in_collision(node_p.current, node_n.current):
                        self.updateVertex(node_p, node_n)

                # goal found
                if node_n.current == self.goal:
                    heapq.heappush(OPEN, node_n)
                    break
                
                # update OPEN list
                heapq.heappush(OPEN, node_n)

            CLOSED[node.current] = node

        self.failed_info[1]["expand"] = CLOSED
        return self.failed_info

    def updateVertex(self, node_p: Node, node_n: Node) -> None:
        """
        Update extend node information with current node's parent node.

        Args:
            node_p (Node): parent node
            node_n (Node): next node
        """
        if node_p.g + self.get_cost(node_p.current, node_n.current) <= node_n.g:
            node_n.g = node_p.g + self.get_cost(node_p.current, node_n.current)
            node_n.parent = node_p.current
