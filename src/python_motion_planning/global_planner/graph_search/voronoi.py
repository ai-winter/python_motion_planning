"""
@file: voronoi.py
@breif: Voronoi-based motion planning
@author: Yang Haodong, Wu Maojia
@update: 2024.6.23
"""
import heapq, math
import numpy as np
from scipy.spatial import cKDTree, Voronoi

from .graph_search import GraphSearcher
from python_motion_planning.utils import Env, Node, Grid

class VoronoiPlanner(GraphSearcher):
    """
    Class for Voronoi-based motion planning.

    Parameters:
        start (tuple): start point coordinate
        goal (tuple): goal point coordinate
        env (Grid): environment
        heuristic_type (str): heuristic function type, default is euclidean
        n_knn (int): number of edges from one sampled point
        max_edge_len (float): maximum edge length
        inflation_r (float): inflation range

    Examples:
        >>> import python_motion_planning as pmp
        >>> planner = pmp.VoronoiPlanner((5, 5), (45, 25), pmp.Grid(51, 31))
        >>> cost, path, _ = planner.plan()     # planning results only
        >>> planner.plot.animation(path, str(planner), cost, expand)  # animation
        >>> planner.run()       # run both planning and animation
    """
    def __init__(self, start: tuple, goal: tuple, env: Grid, heuristic_type: str = "euclidean", \
                 n_knn: int = 10, max_edge_len: float = 10.0, inflation_r: float = 1.0) -> None:
        super().__init__(start, goal, env, heuristic_type)
        # number of edges from one sampled point
        self.n_knn = n_knn
        # maximum edge length
        self.max_edge_len = max_edge_len
        # inflation range
        self.inflation_r = inflation_r

    def __str__(self) -> str:
        return "Voronoi-based Planner"

    def plan(self):
        """
        Voronoi-based motion plan function.

        Returns:
            cost (float): path cost
            path (list): planning path
            expand (list): voronoi sampled nodes
        """
        # sampling voronoi diagram
        vor = Voronoi(np.array(list(self.env.obstacles)))
        vx_list = [ix for [ix, _] in vor.vertices] + [self.start.x, self.goal.x]
        vy_list = [iy for [_, iy] in vor.vertices] + [self.start.y, self.goal.y]
        sample_num = len(vx_list)
        expand = [Node((vx_list[i], vy_list[i])) for i in range(sample_num)]

        # generate road map for voronoi nodes
        road_map = {}
        node_tree = cKDTree(np.vstack((vx_list, vy_list)).T)

        for node in expand:
            edges = []
            _, index_list = node_tree.query([node.x, node.y], k=sample_num)

            for i in range(1, len(index_list)):
                node_n = expand[index_list[i]]

                if not self.isCollision(node, node_n):
                    edges.append(node_n)

                if len(edges) >= self.n_knn:
                    break

            road_map[node] = edges

        # calculate shortest path using graph search algorithm
        cost, path = self.getShortestPath(road_map)
        return cost, path, expand

    def run(self):
        """
        Running both plannig and animation.
        """
        cost, path, expand = self.plan()
        self.plot.animation(path, str(self), cost, expand)
    
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
        # OPEN list (priority queue) and CLOSED list (hash table)
        OPEN = []
        heapq.heappush(OPEN, self.start)
        CLOSED = dict()

        while OPEN:
            node = heapq.heappop(OPEN)

            # exists in CLOSED list
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
        return [], []

    def extractPath(self, closed_list: dict):
        """
        Extract the path based on the CLOSED list.

        Parameters:
            closed_list (dict): CLOSED list

        Returns:
            cost (float): the cost of planning path
            path (list): the planning path
        """
        cost = 0
        node = closed_list[self.goal.current]
        path = [node.current]
        while node != self.start:
            node_parent = closed_list[node.parent]
            cost += self.dist(node, node_parent)
            node = node_parent
            path.append(node.current)
        return cost, path

    def isCollision(self, node1: Node, node2: Node) -> bool:
        """
        Judge collision when moving from node1 to node2.

        Parameters:
            node1 (Node): start node
            node2 (Node): end node

        Returns:
            collision (bool): True if collision exists else False
        """
        if node1.current in self.obstacles or node2.current in self.obstacles:
            return True
        
        yaw = self.angle(node1, node2)
        dist = self.dist(node1, node2)

        if dist >= self.max_edge_len:
            return True

        d_dist = self.inflation_r
        n_step = round(dist / d_dist)

        x, y = node1.current
        for _ in range(n_step):
            dist_to_obs, _ = self.env.obstacles_tree.query([x, y])
            if dist_to_obs <= self.inflation_r:
                return True
            x += d_dist * math.cos(yaw)
            y += d_dist * math.sin(yaw)

        # goal point check
        dist_to_obs, _ = self.env.obstacles_tree.query(node2.current)
        if dist_to_obs <= self.inflation_r:
            return True

        return False