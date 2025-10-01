import math
import random
from typing import Union, Dict, List, Tuple

from python_motion_planning.common import BaseMap, Node
from .rrt import RRT

class RRTStar(RRT):
    """
    Class for RRT* (Rapidly-exploring Random Tree Star) path planner.
    
    RRT* is an optimized version of RRT that provides asymptotically optimal paths
    by rewiring the tree to maintain lower-cost connections.

    Args:
        *args: see the parent class.
        max_dist: Maximum expansion distance for each step (default: 5.0).
        sample_num: Maximum number of samples to generate (default: 100000).
        goal_sample_rate: Probability of sampling the goal directly (default: 0.05).
        radius: Radius for finding nearby nodes during rewiring (default: 10.0).
        *kwargs: see the parent class.

    References:
        [1] Sampling-based algorithms for optimal motion planning
    """
    def __init__(self, *args, 
                 radius: float = 10.0,** kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Radius for finding nearby nodes during rewiring phase
        self.radius = radius

    def __str__(self) -> str:
        return "Rapidly-exploring Random Tree Star (RRT*)"

    def plan(self) -> Union[list, dict]:
        """
        RRT* path planning algorithm implementation with optimality properties.
        
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

            # Find all nearby nodes within radius
            near_nodes = self._find_near_nodes(tree, node_new)
            
            # Select optimal parent from nearby nodes
            node_new, node_near = self._choose_parent(node_new, node_near, near_nodes)
            if node_new is None:
                continue
                
            # Add new node to tree
            tree[node_new.current] = node_new
            
            # Rewire tree to potentially improve existing paths
            self._rewire(tree, node_new, near_nodes)

            # Check if goal is reachable
            dist_to_goal = self.get_cost(node_new.current, self.goal)
            if dist_to_goal <= self.max_dist:
                # Check final edge to goal
                if not self.map_.in_collision(node_new.current, self.goal):
                    goal_cost = node_new.g + dist_to_goal
                    # Update goal node if it already exists with a lower cost path
                    if self.goal in tree:
                        if goal_cost < tree[self.goal].g:
                            tree[self.goal].parent = node_new.current
                            tree[self.goal].g = goal_cost
                    else:
                        goal_node = Node(self.goal, node_new.current, goal_cost, 0)
                        tree[self.goal] = goal_node
                    
                    # Extract and return the path
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

    def _find_near_nodes(self, tree: Dict[tuple, Node], node_new: Node) -> List[Node]:
        """
        Find all nodes in the tree within a specified radius of the new node.
        
        Args:
            tree: Current tree of nodes
            node_new: Newly created node
            
        Returns:
            near_nodes: List of nodes within the specified radius
        """
        near_nodes = []
        for node in tree.values():
            if self.get_cost(node.current, node_new.current) <= self.radius:
                near_nodes.append(node)
        return near_nodes

    def _choose_parent(self, node_new: Node, node_near: Node, near_nodes: List[Node]) -> Tuple[Union[Node, None], Node]:
        """
        Select the optimal parent for the new node from nearby nodes to minimize cost.
        
        Args:
            node_new: Newly created node
            node_near: Nearest node in the tree
            near_nodes: List of nearby nodes
            
        Returns:
            node_new: Updated new node with optimal parent
            node_near: The chosen parent node
        """
        # Initialize with nearest node as potential parent
        node_new.g = node_near.g + self.get_cost(node_near.current, node_new.current)
        best_parent = node_near
        
        # Check all nearby nodes for potentially lower-cost paths
        for node_near_candidate in near_nodes:
            # Skip if candidate is the same as initial nearest node
            if node_near_candidate.current == best_parent.current:
                continue
                
            # Check if path from candidate to new node is collision-free
            if self.map_.in_collision(node_near_candidate.current, node_new.current):
                continue
                
            # Calculate cost through this candidate parent
            new_cost = node_near_candidate.g + self.get_cost(node_near_candidate.current, node_new.current)
            
            # Update parent if this path is cheaper
            if new_cost < node_new.g:
                node_new.g = new_cost
                best_parent = node_near_candidate
        
        # Set the best parent for the new node
        node_new.parent = best_parent.current
        return node_new, best_parent

    def _rewire(self, tree: Dict[tuple, Node], node_new: Node, near_nodes: List[Node]) -> None:
        """
        Rewire the tree to potentially improve paths for existing nodes using the new node.
        
        Args:
            tree: Current tree of nodes
            node_new: Newly added node
            near_nodes: List of nearby nodes
        """
        for node_near in near_nodes:
            # Skip if node is the parent of the new node
            if node_near.current == node_new.parent:
                continue
                
            # Check if path from new node to existing node is collision-free
            if self.map_.in_collision(node_new.current, node_near.current):
                continue
                
            # Calculate potential new cost for existing node through new node
            new_cost = node_new.g + self.get_cost(node_new.current, node_near.current)
            
            # Update parent if new path is cheaper
            if new_cost < node_near.g:
                node_near.parent = node_new.current
                node_near.g = new_cost
                # Update the node in the tree
                tree[node_near.current] = node_near
