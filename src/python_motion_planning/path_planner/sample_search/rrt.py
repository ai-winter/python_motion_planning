"""
@file: rrt.py
@author: Wu Maojia, Yang Haodong
@update: 2025.12.19
"""
import math
import random
from typing import Union, Dict, List, Tuple, Any

import numpy as np
import faiss

from python_motion_planning.common import BaseMap, Node, TYPES, Grid
from python_motion_planning.path_planner import BasePathPlanner

class RRT(BasePathPlanner):
    """
    Class for RRT (Rapidly-exploring Random Tree) path planner.

    Args:
        *args: see the parent class.
        max_dist: Maximum expansion distance for each step.
        sample_num: Maximum number of samples to generate.
        goal_sample_rate: Probability of sampling the goal directly.
        discrete: Whether to use discrete or continuous space.
        faiss: Whether to use Faiss to accelerate the search.
        *kwargs: see the parent class.

    References:
        [1] Rapidly-Exploring Random Trees: A New Tool for Path Planning

    Examples:
        >>> map_ = Grid(bounds=[[0, 15], [0, 15]])
        >>> planner = RRT(map_=map_, start=(5, 5), goal=(10, 10))
        >>> path, path_info = planner.plan()
        >>> print(path_info['success'])
        True
        
        >>> planner.map_.type_map[3:10, 6] = TYPES.OBSTACLE
        >>> path, path_info = planner.plan()
        >>> print(path_info['success'])
        True
    """
    def __init__(self, *args, 
                 max_dist: float = 5.0, sample_num: int = 100000, 
                 goal_sample_rate: float = 0.05,
                 discrete: bool = False,
                 use_faiss: bool = True,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_dist = max_dist
        self.sample_num = sample_num
        self.goal_sample_rate = goal_sample_rate
        self.discrete = discrete
        self.use_faiss = use_faiss

    def __str__(self) -> str:
        return "RRT"

    def plan(self) -> Union[List[Tuple[float, ...]], Dict[str, Any]]:
        """
        RRT path planning algorithm implementation.

        Returns:
            path: A list containing the path waypoints
            path_info: A dictionary containing path information
        """
        # Initialize tree with start node
        tree = {}
        start_node = Node(self.start, None, 0, 0)
        tree[self.start] = start_node

        # Initialize FAISS index
        if self.use_faiss:
            faiss_index = faiss.IndexFlatL2(self.dim)
            faiss_nodes = []
            self._faiss_add_node(start_node, faiss_index, faiss_nodes)

        # Main sampling loop
        for _ in range(self.sample_num):
            # Generate random sample node
            node_rand = self._generate_random_node()
            
            # Skip if node already exists
            if node_rand.current in tree:
                continue
                
            # Find nearest node in tree
            node_near = self._get_nearest_node(tree, node_rand, faiss_index, faiss_nodes)
            
            # Create new node towards random sample
            node_new = self._steer(node_near, node_rand)
            if node_new is None:
                continue
                
            # Check if edge is collision-free
            if self.map_.in_collision(
                self.map_.point_float_to_int(node_new.current), 
                self.map_.point_float_to_int(node_near.current)
                ):
                continue

            # Add new node to tree
            node_new.parent = node_near.current
            node_new.g = node_near.g + self.get_cost(node_near.current, node_new.current)
            tree[node_new.current] = node_new
            if self.use_faiss:
                self._faiss_add_node(node_new, faiss_index, faiss_nodes)

            # Check if goal is reachable
            dist_to_goal = self.get_cost(node_new.current, self.goal)
            if dist_to_goal <= self.max_dist:
                # Check final edge to goal
                if not self.map_.in_collision(
                    self.map_.point_float_to_int(node_new.current), 
                    self.map_.point_float_to_int(self.goal)
                    ):
                    if node_new.current == self.goal:
                        goal_node = node_new
                    else:
                        goal_node = Node(self.goal, node_new.current, 
                                        node_new.g + dist_to_goal, 0)
                    tree[self.goal] = goal_node
                    path, length, cost = self.extract_path(tree)
                    return path, {
                        "success": True,
                        "start": self.start,
                        "goal": self.goal,
                        "length": length,
                        "cost": cost,
                        "expand": tree,
                    }

        # Planning failed
        self.failed_info[1]["expand"] = tree
        return self.failed_info

    def _generate_random_node(self) -> Node:
        """
        Generate a random node within map bounds as integer grid point.

        Returns:
            node: Generated random node on grid
        """
        # Sample goal directly with specified probability
        if random.random() < self.goal_sample_rate:
            return Node(self.goal, None, 0, 0)
            
        point = []
        # Generate random integer point within grid bounds
        for d in range(self.dim):
            d_min, d_max = -0.5, self.map_.shape[d] - 0.5
            point.append(random.uniform(d_min, d_max))
        point = tuple(point)

        if self.discrete:
            point = self.map_.point_float_to_int(point)

        return Node(point, None, 0, 0)

    def _get_nearest_node(self, tree: Dict[Tuple[int, ...], Node], 
                         node_rand: Node, index=None, nodes=None) -> Node:
        """
        Find the nearest node in the tree to a random sample.

        Args:
            tree: Current tree of nodes
            node_rand: Random sample node
            index: FAISS index (required when `use_faiss`=True)
            nodes: List of nodes in FAISS index (required when `use_faiss`=True)

        Returns:
            node: Nearest node in the tree
        """
        # knn search using faiss
        if self.use_faiss:
            query = np.array(node_rand.current, dtype=np.float32).reshape(1, -1)
            _, indices = index.search(query, 1)
            return nodes[indices[0][0]]

        # brute force search
        min_dist = float('inf')
        nearest_node = None
        
        for node in tree.values():
            dist = self.get_cost(node.current, node_rand.current)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
                
        return nearest_node

    def _steer(self, node_near: Node, 
              node_rand: Node) -> Union[Node, None]:
        """
        Steer from nearest node towards random sample.

        Args:
            node_near: Nearest node in tree
            node_rand: Random sample node

        Returns:
            node: New node in direction of random sample
        """
        # Calculate differences for each dimension
        diffs = [node_rand.current[i] - node_near.current[i] for i in range(self.dim)]
        
        # Calculate Euclidean distance in n-dimensional space
        dist = math.sqrt(sum(diff**2 for diff in diffs))
        
        # Handle case where nodes are coincident
        if math.isclose(dist, 0):
            return None
            
        # If within max distance, use the random node directly
        if dist <= self.max_dist:
            return node_rand
            
        # Otherwise scale to maximum distance
        scale = self.max_dist / dist
        new_point = [
            node_near.current[i] + scale * diffs[i]
            for i in range(self.dim)
        ]
        new_point = tuple(new_point)

        if self.discrete:
            new_point = self.map_.point_float_to_int(new_point)
            
        return Node(new_point, None, 0, 0)

    def _faiss_add_node(self, node: Node, index, nodes):
        """
        Add a node to the FAISS index.

        Args:
            node: Node to add
            index: FAISS index
            nodes: List of nodes in FAISS index
        """
        vec = np.array(node.current, dtype=np.float32).reshape(1, -1)
        index.add(vec)
        nodes.append(node)
