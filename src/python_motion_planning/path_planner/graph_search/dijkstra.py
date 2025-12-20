"""
@file: dijkstra.py
@author: Wu Maojia
@update: 2025.10.6
"""
from typing import Union, List, Tuple, Dict, Any
import heapq
 
from python_motion_planning.common import BaseMap, Grid, Node, TYPES
from python_motion_planning.path_planner import BasePathPlanner

class Dijkstra(BasePathPlanner):
    """
    Class for Dijkstra path planner.

    Args:
        *args: see the parent class.
        diagonal: whether to allow diagonal expansions.
        *kwargs: see the parent class.

    References:
        [1] A Note on Two Problems in Connexion with Graphs

    Examples:
        >>> map_ = Grid(bounds=[[0, 15], [0, 15]])
        >>> planner = Dijkstra(map_=map_, start=(5, 5), goal=(10, 10))
        >>> path, path_info = planner.plan()
        >>> print(path_info['success'])
        True
        
        >>> planner.map_.type_map[3:10, 6] = TYPES.OBSTACLE
        >>> path, path_info = planner.plan()
        >>> print(path_info['success'])
        True
    """
    def __init__(self, *args, diagonal: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.diagonal = diagonal

    def __str__(self) -> str:
        return "Dijkstra"

    def plan(self) -> Union[List[Tuple[float, ...]], Dict[str, Any]]:
        """
        Interface for planning.

        Returns:
            path: A list containing the path waypoints
            path_info: A dictionary containing the path information
        """
        # OPEN list (priority queue) and CLOSED list (hash table)
        OPEN = []
        # For Dijkstra, we only use g-value (no heuristic h-value)
        start_node = Node(self.start, None, 0, 0)
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

                # For Dijkstra, we only update g-value (no heuristic)
                node_n.g = node.g + self.get_cost(node.current, node_n.current)

                # goal found
                if node_n.current == self.goal:
                    heapq.heappush(OPEN, node_n)
                    break
                
                # update OPEN list with node sorted by g-value
                heapq.heappush(OPEN, node_n)

            CLOSED[node.current] = node

        self.failed_info[1]["expand"] = CLOSED
        return self.failed_info
