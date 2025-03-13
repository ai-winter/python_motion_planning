"""
@file: voronoi_planner.py
@breif: Voronoi-based path planning
@author: Yang Haodong, Wu Maojia
@update: 2024.2.11
"""
import heapq, math
import numpy as np

from scipy.spatial import cKDTree, Voronoi
from typing import List, Tuple, Dict

from python_motion_planning.path_planner import PathPlanner
from python_motion_planning.common.utils import LOG
from python_motion_planning.common.structure import Node, Env
from python_motion_planning.common.geometry import Point3d

class VoronoiPlanner(PathPlanner):
    """
    Class for Voronoi-based path planning.

    Parameters:
        env (Env): environment object
        params (dict): parameters
    """
    def __init__(self, env: Env, params: dict) -> None:
        super().__init__(env, params)

    def __str__(self) -> str:
        return "Voronoi-based PathPlanner"

    def plan(self, start: Point3d, goal: Point3d) -> Tuple[List[Point3d], List[Dict]]:
        """
        Voronoi-based motion plan function.

        Parameters:
            start (Point3d): The starting point of the planning path.
            goal (Point3d): The goal point of the planning path.

        Returns:
            path (List[Point3d]): The planned path from start to goal.
            visual_info (List[Dict]): Information for visualization
        """
        self.start = Node(start, start, 0, 0)
        self.goal = Node(goal, goal, 0, 0)

        # sampling voronoi diagram
        vor = Voronoi(np.array(self.obstacles))
        vx_list = [ix for [ix, _] in vor.vertices] + [self.start.x, self.goal.x]
        vy_list = [iy for [_, iy] in vor.vertices] + [self.start.y, self.goal.y]
        sample_num = len(vx_list)
        expand = [Node(Point3d(ix, iy)) for [ix, iy] in vor.vertices] + [self.start, self.goal]

        # generate road map for voronoi nodes
        road_map = {}
        node_tree = cKDTree(np.vstack((vx_list, vy_list)).T)

        for node in expand:
            edges = []
            _, index_list = node_tree.query([node.x, node.y], k=sample_num)

            for i in range(1, len(index_list)):
                node_n = expand[index_list[i]]
                if not self.collision_checker(node.current, node_n.current):
                    edges.append(node_n)

                if len(edges) >= self.n_knn:
                    break

            road_map[node] = edges

        # calculate shortest path using graph search algorithm
        cost, path = self.getShortestPath(road_map)

        if path != []:
            LOG.INFO(f"{str(self)} PathPlanner Planning Successfully. Cost: {cost}")
            return path, [
                {"type": "value", "data": True, "name": "success"},
                {"type": "value", "data": cost, "name": "cost"},
                {"type": "path", "data": path, "name": "normal"},
                {"type": "grids", "name": "expand",
                "data": [n.current for n in expand if (int(n.x), int(n.y)) not in self.obstacles]}
            ]
        else:
            LOG.INFO(f"Planning Failed.")
            return path, [
                {"type": "value", "data": False, "name": "success"},
                {"type": "value", "data": 0, "name": "cost"},
                {"type": "path", "data": [], "name": "normal"},
                {"type": "grids", "name": "expand",
                "data": [n.current for n in expand if (int(n.x), int(n.y)) not in self.obstacles]}
            ]
    
    def getShortestPath(self, road_map: dict, dijkstra: bool = True) -> list:
        """
        Calculate shortest path using graph search algorithm(A*, Dijkstra, etc).

        Parameters:
            road_map (dict): road map for voronoi diagram, which store KNN for one voronoi node
            dijkstra (bool): using Dijkstra if true, else A*

        Returns:
            cost (float): path cost
            path (list): planning path
        """
        # OPEN set with priority and CLOSED set
        OPEN = []
        heapq.heappush(OPEN, self.start)
        CLOSED = dict()

        while OPEN:
            node = heapq.heappop(OPEN)

            # exists in CLOSED set
            if node.current in CLOSED:
                continue

            # goal found
            if node == self.goal:
                CLOSED[node.current] = node
                return self.extractPath(CLOSED)

            for node_n in road_map[node]:                
                # exists in CLOSED set
                if node_n.current in CLOSED:
                    continue
                
                node_n.parent = node.current
                node_n.g = self.dist(node_n, node)
                if not dijkstra:
                    node_n.h = self.h(node_n, self.goal)

                # goal found
                if node_n == self.goal:
                    heapq.heappush(OPEN, node_n)
                    break
                
                # update OPEN set
                heapq.heappush(OPEN, node_n)
            
            CLOSED[node.current] = node
        return ([], [])

    def extractPath(self, closed_set: dict):
        """
        Extract the path based on the CLOSED set.

        Parameters:
            closed_set (list): CLOSED set

        Returns:
            cost (float): the cost of planning path
            path (list): the planning path
        """
        cost = 0
        node = closed_set[self.goal.current]
        path = [node.current]
        while node != self.start:
            node_parent = closed_set[node.parent]
            cost += self.dist(node, node_parent)
            node = node_parent
            path.append(node.current)
        path = list({}.fromkeys(path).keys())
        return cost, list(reversed(path))
    
    def h(self, node: Node, goal: Node) -> float:
        """
        Calculate heuristic.

        Parameters:
            node (Node): current node
            goal (Node): goal node

        Returns:
            h (float): heuristic function value of node
        """
        if self.heuristic_type == "manhattan":
            return abs(goal.x - node.x) + abs(goal.y - node.y)
        elif self.heuristic_type == "euclidean":
            return math.hypot(goal.x - node.x, goal.y - node.y)