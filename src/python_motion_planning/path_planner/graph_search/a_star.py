"""
@file: a_star.py
@breif: A* planner
@author: Wu Maojia
@update: 2025.9.5
"""
from typing import Union
import heapq
 
from python_motion_planning.common import BaseMap, Grid, Node, TYPES
from python_motion_planning.path_planner import BasePathPlanner

class AStar(BasePathPlanner):
    """
    Class for building path planner.

    Parameters:
        map_: The map which the planner is based on.
        start: The start point of the planner in the map coordinate system.
        goal: The goal point of the planner in the map coordinate system.

    Examples:
        >>> map_ = Grid(bounds=[[0, 15], [0, 15]])
        >>> planner = AStar(map_=map_, start=(5, 5), goal=(10, 10))
        >>> planner.plan()
        ([(5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)], {'success': True, 'start': (5, 5), 'goal': (10, 10), 'length': 7.0710678118654755, 'cost': 7.0710678118654755, 'expand': [Node((5, 5), None, 0, 7.0710678118654755), Node((6, 6), (5, 5), 1.4142135623730951, 5.656854249492381), Node((7, 7), (6, 6), 2.8284271247461903, 4.242640687119285), Node((8, 8), (7, 7), 4.242640687119286, 2.8284271247461903), Node((9, 9), (8, 8), 5.656854249492381, 1.4142135623730951), Node((10, 10), (9, 9), 7.0710678118654755, 0.0)]})

        >>> planner.map_.type_map[3:10, 6] = TYPES.OBSTACLE
        >>> planner.plan()
        ([(5, 5), (6, 5), (7, 5), (8, 5), (9, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10)], {'success': True, 'start': (5, 5), 'goal': (10, 10), 'length': 9.414213562373096, 'cost': 9.414213562373096, 'expand': [Node((5, 5), None, 0, 7.0710678118654755), Node((6, 5), (5, 5), 1.0, 6.4031242374328485), Node((7, 5), (6, 5), 2.0, 5.830951894845301), Node((8, 5), (7, 5), 3.0, 5.385164807134504), Node((6, 4), (5, 5), 1.4142135623730951, 7.211102550927978), Node((5, 4), (5, 5), 1.0, 7.810249675906654), Node((4, 5), (5, 5), 1.0, 7.810249675906654), Node((9, 5), (8, 5), 4.0, 5.0990195135927845), Node((7, 4), (6, 5), 2.414213562373095, 6.708203932499369), Node((10, 6), (9, 5), 5.414213562373095, 4.0), Node((10, 7), (10, 6), 6.414213562373095, 3.0), Node((10, 8), (10, 7), 7.414213562373095, 2.0), Node((10, 9), (10, 8), 8.414213562373096, 1.0), Node((10, 10), (10, 9), 9.414213562373096, 0.0)]})

        >>> AStar(map_=map_, start=(6, 6), goal=(10, 10)).plan()
        ([], {'success': False, 'start': None, 'goal': None, 'length': 0, 'cost': 0, 'expand': []})

    References:
        [1] A Formal Basis for the heuristic Determination of Minimum Cost Paths
    """
    def __init__(self, map_: BaseMap, start: tuple, goal: tuple) -> None:
        super().__init__(map_, start, goal)
        self.start = start
        self.goal = goal

    def __str__(self) -> str:
        return "A*"

    def get_cost(self, p1: tuple, p2: tuple) -> float:
        """
        Get the cost between two points. (default: distance defined in the map)

        Parameters:
            p1: Start point.
            p2: Goal point.
        
        Returns:
            cost: Cost between two points.
        """
        return self.map_.get_distance(p1, p2)

    def get_heuristic(self, point: tuple) -> float:
        """
        Get the heuristic value of the point. (default: cost between current point and goal point)

        Parameters:
            p1: Start point.
            p2: Goal point.
        
        Returns:
            heuristic: Heuristic value of the point.
        """
        return self.get_cost(point, self.goal)

    def plan(self) -> Union[list, dict]:
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

            # obstacle found
            if not self.map_.is_expandable(node.current):
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
                    "expand": list(CLOSED.values())
                }

            for node_n in self.map_.get_neighbors(node): 
                # exists in CLOSED list
                if node_n.current in CLOSED:
                    continue

                node_n.g = node.g + self.get_cost(node.current, node_n.current)
                node_n.h = self.get_heuristic(node_n.current)

                # goal found
                if node_n.current == self.goal:
                    heapq.heappush(OPEN, node_n)
                    break
                
                # update OPEN list
                heapq.heappush(OPEN, node_n)

            CLOSED[node.current] = node

        self.failed_info[1]["expand"] = list(CLOSED.values())
        return self.failed_info

    
    def extract_path(self, closed_list: dict) -> tuple:
        """
        Extract the path based on the CLOSED list.

        Parameters:
            closed_list (dict): CLOSED list

        Returns:
            cost (float): the cost of planned path
            path (list): the planning path
        """
        length = 0
        cost = 0
        node = closed_list[self.goal]
        path = [node.current]
        while node.current != self.start:
            node_parent = closed_list[node.parent]
            length += self.map_.get_distance(node.current, node_parent.current)
            cost += self.get_cost(node.current, node_parent.current)
            node = node_parent
            path.append(node.current)
        path = path[::-1]   # make the order: start -> goal
        return path, length, cost