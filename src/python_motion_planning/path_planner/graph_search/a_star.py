"""
@file: a_star.py
@breif: A* planner
@author: Wu Maojia
@update: 2025.3.29
"""
from typing import Union
import heapq
 
from python_motion_planning.common import Map, Grid, Point2D, PointND, Node, TYPES
from python_motion_planning.path_planner import PathPlanner

class AStar(PathPlanner):
    """
    Class for building path planner.

    Parameters:
        map_: The map which the planner is based on.

    Examples:
        >>> map_ = Grid(bounds=[[0, 15], [0, 15]])
        >>> planner = AStar(map_=map_)
        >>> planner.plan(start=Point2D(5, 5), goal=Point2D(10, 10))
        ([Point2D([5, 5]), PointND([6, 6]), PointND([7, 7]), PointND([8, 8]), PointND([9, 9]), PointND([10, 10])], {'success': True, 'start': Point2D([5, 5]), 'goal': Point2D([10, 10]), 'length': 7.0710678118654755, 'cost': 7.0710678118654755, 'expand': [Node(Point2D([5, 5]), None, 0, 7.0710678118654755), Node(PointND([6, 6]), Point2D([5, 5]), 1.4142135623730951, 5.656854249492381), Node(PointND([7, 7]), PointND([6, 6]), 2.8284271247461903, 4.242640687119285), Node(PointND([8, 8]), PointND([7, 7]), 4.242640687119286, 2.8284271247461903), Node(PointND([9, 9]), PointND([8, 8]), 5.656854249492381, 1.4142135623730951), Node(PointND([10, 10]), PointND([9, 9]), 7.0710678118654755, 0.0)]})
        
        >>> planner.map_.type_map[3:10, 6] = TYPES.OBSTACLE
        >>> planner.plan(start=Point2D(5, 5), goal=Point2D(10, 10))
        ([Point2D([5, 5]), PointND([6, 5]), PointND([7, 5]), PointND([8, 5]), PointND([9, 5]), PointND([10, 6]), PointND([10, 7]), PointND([10, 8]), PointND([10, 9]), PointND([10, 10])], {'success': True, 'start': Point2D([5, 5]), 'goal': Point2D([10, 10]), 'length': 9.414213562373096, 'cost': 9.414213562373096, 'expand': [Node(Point2D([5, 5]), None, 0, 7.0710678118654755), Node(PointND([6, 5]), Point2D([5, 5]), 1.0, 6.4031242374328485), Node(PointND([7, 5]), PointND([6, 5]), 2.0, 5.830951894845301), Node(PointND([8, 5]), PointND([7, 5]), 3.0, 5.385164807134504), Node(PointND([6, 4]), Point2D([5, 5]), 1.4142135623730951, 7.211102550927978), Node(PointND([5, 4]), Point2D([5, 5]), 1.0, 7.810249675906654), Node(PointND([4, 5]), Point2D([5, 5]), 1.0, 7.810249675906654), Node(PointND([9, 5]), PointND([8, 5]), 4.0, 5.0990195135927845), Node(PointND([7, 4]), PointND([6, 5]), 2.414213562373095, 6.708203932499369), Node(PointND([10, 6]), PointND([9, 5]), 5.414213562373095, 4.0), Node(PointND([10, 7]), PointND([10, 6]), 6.414213562373095, 3.0), Node(PointND([10, 8]), PointND([10, 7]), 7.414213562373095, 2.0), Node(PointND([10, 9]), PointND([10, 8]), 8.414213562373096, 1.0), Node(PointND([10, 10]), PointND([10, 9]), 9.414213562373096, 0.0)]})

        >>> planner.plan(start=Point2D(6, 6), goal=Point2D(10, 10))
        ([], {'success': False, 'start': None, 'goal': None, 'length': 0, 'cost': 0, 'expand': []})
    """
    def __init__(self, map_: Map) -> None:
        super().__init__(map_)
        self.start = None
        self.goal = None

    def __str__(self) -> str:
        return "A*"

    def getCost(self, p1: PointND, p2: PointND) -> float:
        """
        Get the cost between two points. (default: distance defined in the map)

        Parameters:
            p1: Start point.
            p2: Goal point.
        
        Returns:
            cost: Cost between two points.
        """
        return self.map_.getDistance(p1, p2)

    def getHeuristic(self, point: PointND) -> float:
        """
        Get the heuristic value of the point. (default: cost between current point and goal point)

        Parameters:
            p1: Start point.
            p2: Goal point.
        
        Returns:
            heuristic: Heuristic value of the point.
        """
        return self.getCost(point, self.goal)

    def plan(self, start: PointND, goal: PointND) -> Union[list, dict]:
        """
        Interface for planning.

        Parameters:
            start: Start point
            goal: Goal point

        Returns:
            path: A list containing the path waypoints
            path_info: A dictionary containing the path information (success, length, cost, expand)
        """
        start = start.astype(self.map_.dtype)
        goal = goal.astype(self.map_.dtype)

        self.start = start
        self.goal = goal

        # OPEN list (priority queue) and CLOSED list (hash table)
        OPEN = []
        start_node = Node(start, None, 0, self.getHeuristic(start))
        heapq.heappush(OPEN, start_node)
        CLOSED = dict()

        while OPEN:
            node = heapq.heappop(OPEN)

            # obstacle found
            if self.map_.type_map[tuple(node.current)] == TYPES.OBSTACLE:
                continue

            # exists in CLOSED list
            if node.current in CLOSED:
                continue

            # goal found
            if node.current == self.goal:
                CLOSED[node.current] = node
                path, length, cost = self.extractPath(CLOSED)
                return path, {
                    "success": True, 
                    "start": self.start, 
                    "goal": self.goal, 
                    "length": length, 
                    "cost": cost, 
                    "expand": list(CLOSED.values())
                }

            for node_n in self.map_.getNeighbor(node, cost_func=self.getCost, heuristic_func=self.getHeuristic): 
                # exists in CLOSED list
                if node_n.current in CLOSED:
                    continue

                # goal found
                if node_n.current == self.goal:
                    heapq.heappush(OPEN, node_n)
                    break
                
                # update OPEN list
                heapq.heappush(OPEN, node_n)

            CLOSED[node.current] = node

        return self.failed_info

    
    def extractPath(self, closed_list: dict) -> tuple:
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
            length += self.map_.getDistance(node.current, node_parent.current)
            cost += self.getCost(node.current, node_parent.current)
            node = node_parent
            path.append(node.current)
        path = path[::-1]   # make the order: start -> goal
        return path, length, cost