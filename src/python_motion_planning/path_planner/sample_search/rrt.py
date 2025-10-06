"""
@file: rrt.py
@author: Wu Maojia, Yang Haodong
@update: 2025.10.6
"""
import math
import random
import numpy as np
from typing import Union, Dict, List, Tuple, Any

from python_motion_planning.common import BaseMap, Node, TYPES, Grid
from python_motion_planning.path_planner import BasePathPlanner

class RRT(BasePathPlanner):
    """
    Class for RRT (Rapidly-exploring Random Tree) path planner.

    Args:
        *args: see the parent class.
        max_dist: Maximum expansion distance for each step (default: 1.0).
        sample_num: Maximum number of samples to generate (default: 10000).
        goal_sample_rate: Probability of sampling the goal directly (default: 0.05).
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
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Maximum expansion distance per step
        self.max_dist = max_dist
        # Maximum number of samples
        self.sample_num = sample_num
        # Goal sampling probability
        self.goal_sample_rate = goal_sample_rate
        # Environment bounds from the map
        self.bounds = self.map_.bounds

    def __str__(self) -> str:
        return "Rapidly-exploring Random Tree (RRT)"

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

        # Main sampling loop
        for _ in range(self.sample_num):
            # Generate random sample node
            node_rand = self._generate_random_node()
            
            # Skip if node already exists
            if node_rand.current in tree:
                continue
                
            # Find nearest node in tree
            node_near = self._get_nearest_node(tree, node_rand)
            
            # Create new node towards random sample
            node_new = self._steer(node_near, node_rand)
            if node_new is None:
                continue
                
            # Check if edge is collision-free
            if self.map_.in_collision(node_new.current, node_near.current):
                continue

            # Add new node to tree
            node_new.parent = node_near.current
            node_new.g = node_near.g + self.get_cost(node_near.current, node_new.current)
            tree[node_new.current] = node_new

            # Check if goal is reachable
            dist_to_goal = self.get_cost(node_new.current, self.goal)
            if dist_to_goal <= self.max_dist:
                # Check final edge to goal
                if not self.map_.in_collision(node_new.current, self.goal):
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
            d_min, d_max = self.bounds[d]
            point.append(random.randint(int(d_min), int(d_max)))
        point = tuple(point)

        return Node(point, None, 0, 0)

    def _get_nearest_node(self, tree: Dict[Tuple[int, ...], Node], 
                         node_rand: Node) -> Node:
        """
        Find the nearest node in the tree to a random sample.

        Args:
            tree: Current tree of nodes
            node_rand: Random sample node

        Returns:
            node: Nearest node in the tree
        """
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
        new_coords = [
            node_near.current[i] + scale * diffs[i] 
            for i in range(self.dim)
        ]
        
        # Round coordinates if original points were integers
        if all(isinstance(coord, int) for coord in node_near.current):
            new_coords = [round(coord) for coord in new_coords]
            
        return Node(tuple(new_coords), None, 0, 0)