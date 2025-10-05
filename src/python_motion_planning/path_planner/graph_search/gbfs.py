"""
@file: gbfs.py
@author: Wu Maojia
@update: 2025.10.5
"""
from typing import Union
import heapq

from python_motion_planning.common import BaseMap, Grid, Node, TYPES
from python_motion_planning.path_planner.graph_search.dijkstra import Dijkstra

class GBFS(Dijkstra):
    """
    Class for Greedy Best-First Search (GBFS) path planner.

    Args:
        *args: see the parent class.
        *kwargs: see the parent class.

    References:
        [1] Heuristics: Intelligent Search Strategies for Computer Problem Solving.

    Examples:
        >>> map_ = Grid(bounds=[[0, 15], [0, 15]])
        >>> planner = GBFS(map_=map_, start=(5, 5), goal=(10, 10))
        >>> planner.plan()
        ([(5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)], {'success': True, 'start': (5, 5), 'goal': (10, 10), 'length': 7.0710678118654755, 'cost': 7.0710678118654755, 'expand': {(5, 5): Node((5, 5), None, 0, 7.0710678118654755), (6, 6): Node((6, 6), (5, 5), 1.4142135623730951, 5.656854249492381), (7, 7): Node((7, 7), (6, 6), 2.8284271247461903, 4.242640687119285), (8, 8): Node((8, 8), (7, 7), 4.242640687119286, 2.8284271247461903), (9, 9): Node((9, 9), (8, 8), 5.656854249492381, 1.4142135623730951), (10, 10): Node((10, 10), (9, 9), 7.0710678118654755, 0.0)}})
        
        >>> planner.map_.type_map[3:10, 6] = TYPES.OBSTACLE
        >>> planner.plan()
        ([(5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)], {'success': True, 'start': (5, 5), 'goal': (10, 10), 'length': 7.0710678118654755, 'cost': 7.0710678118654755, 'expand': {(5, 5): Node((5, 5), None, 0, 7.0710678118654755), (6, 6): Node((6, 6), (5, 5), 1.4142135623730951, 5.656854249492381), (7, 7): Node((7, 7), (6, 6), 2.8284271247461903, 4.242640687119285), (8, 8): Node((8, 8), (7, 7), 4.242640687119286, 2.8284271247461903), (9, 9): Node((9, 9), (8, 8), 5.656854249492381, 1.4142135623730951), (10, 10): Node((10, 10), (9, 9), 7.0710678118654755, 0.0)}})
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __str__(self) -> str:
        return "GBFS"

    def plan(self) -> Union[list, dict]:
        """
        Interface for planning.

        Returns:
            path: A list containing the path waypoints
            path_info: A dictionary containing the path information (success, length, cost, expand)
        """
        # OPEN list (priority queue) and CLOSED list (hash table)
        OPEN = []
        # For GBFS, we only use h-value (ignore g-value)
        start_node = Node(self.start, None, 0, self.get_heuristic(self.start))
        heapq.heappush(OPEN, start_node)
        CLOSED = dict()

        while OPEN:
            node = heapq.heappop(OPEN)

            # obstacle found
            if not self.map_.is_expandable(node.current, node.parent):
                continue

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

                # For GBFS, we only set h-value
                node_n.g = node.g + self.get_cost(node.current, node_n.current)
                node_n.h = self.get_heuristic(node_n.current)

                # goal found
                if node_n.current == self.goal:
                    heapq.heappush(OPEN, node_n)
                    break

                # update OPEN list with node sorted by h-value
                heapq.heappush(OPEN, node_n)

            CLOSED[node.current] = node

        self.failed_info[1]["expand"] = CLOSED
        return self.failed_info
