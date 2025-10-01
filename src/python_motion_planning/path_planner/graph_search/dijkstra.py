from typing import Union
import heapq
 
from python_motion_planning.common import BaseMap, Grid, Node, TYPES
from python_motion_planning.path_planner import BasePathPlanner

class Dijkstra(BasePathPlanner):
    """
    Class for Dijkstra path planner.

    Args:
        *args: see the parent class.
        *kwargs: see the parent class.

    References:
        [1] A Note on Two Problems in Connexion with Graphs

    Examples:
        >>> map_ = Grid(bounds=[[0, 15], [0, 15]])
        >>> planner = Dijkstra(map_=map_, start=(5, 5), goal=(10, 10))
        >>> planner.plan()
        ([(5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)], {'success': True, 'start': (5, 5), 'goal': (10, 10), 'length': 7.0710678118654755, 'cost': 7.0710678118654755, 'expand': [Node((5, 5), None, 0, 0), Node((6, 6), (5, 5), 1.4142135623730951, 0), ...]})

        >>> planner.map_.type_map[3:10, 6] = TYPES.OBSTACLE
        >>> planner.plan()
        ([(5, 5), (6, 5), (7, 5), (8, 5), (9, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10)], {'success': True, 'start': (5, 5), 'goal': (10, 10), 'length': 9.414213562373096, 'cost': 9.414213562373096, 'expand': [Node((5, 5), None, 0, 0), ...]})

        >>> Dijkstra(map_=map_, start=(6, 6), goal=(10, 10)).plan()
        ([], {'success': False, 'start': None, 'goal': None, 'length': 0, 'cost': 0, 'expand': []})
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __str__(self) -> str:
        return "Dijkstra"

    def plan(self) -> Union[list, dict]:
        """
        Interface for planning.

        Returns:
            path: A list containing the path waypoints
            path_info: A dictionary containing the path information (success, length, cost, expand)
        """
        # OPEN list (priority queue) and CLOSED list (hash table)
        OPEN = []
        # For Dijkstra, we only use g-value (no heuristic h-value)
        start_node = Node(self.start, None, 0, 0)
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

            for node_n in self.map_.get_neighbors(node): 
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
