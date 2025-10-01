import math
import random
import numpy as np
from typing import Union, Dict, List, Tuple

from python_motion_planning.common import BaseMap, Node
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

    def plan(self) -> Union[list, dict]:
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
            
        # Generate random integer point within grid bounds
        x_min, x_max = self.bounds[0]
        y_min, y_max = self.bounds[1]
        x = random.randint(int(x_min), int(x_max))
        y = random.randint(int(y_min), int(y_max))
        return Node((x, y), None, 0, 0)

    def _get_nearest_node(self, tree: Dict[tuple, Node], 
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
        # Calculate direction vector
        dx = node_rand.current[0] - node_near.current[0]
        dy = node_rand.current[1] - node_near.current[1]
        dist = math.hypot(dx, dy)
        
        # Handle case where nodes are coincident
        if math.isclose(dist, 0):
            return None
            
        # Create new node within max distance
        if dist <= self.max_dist:
            return node_rand
            
        # Scale vector to max distance
        scale = self.max_dist / dist
        new_x = node_near.current[0] + scale * dx
        new_y = node_near.current[1] + scale * dy
        new_x, new_y = round(new_x), round(new_y)
        return Node((new_x, new_y), None, 0, 0)