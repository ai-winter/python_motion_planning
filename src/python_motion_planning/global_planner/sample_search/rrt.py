from typing import Union
import random
import math

from python_motion_planning.common import BaseMap, Node
from python_motion_planning.path_planner import BasePathPlanner

class RRT(BasePathPlanner):
    """
    Class for RRT (Rapidly-exploring Random Tree) path planner.

    Args:
        map_: The map which the planner is based on.
        start: The start point of the planner in the map coordinate system.
        goal: The goal point of the planner in the map coordinate system.
        max_dist: Maximum distance for expanding the tree in one step.
        sample_num: Maximum number of samples to generate.
        goal_sample_rate: Probability of sampling the goal point directly.

    References:
        [1] Rapidly-Exploring Random Trees: A New Tool for Path Planning
    """
    def __init__(self, map_: BaseMap, start: tuple, goal: tuple, 
                 max_dist: int = 1, sample_num: int = 10000, 
                 goal_sample_rate: float = 0.05) -> None:
        super().__init__(map_, start, goal)
        self.max_dist = max_dist  # Maximum expansion distance per step
        self.sample_num = sample_num  # Maximum number of samples
        self.goal_sample_rate = goal_sample_rate  # Goal bias probability
        self.tree = {}  # Dictionary to store the tree nodes

    def __str__(self) -> str:
        return "RRT"

    def plan(self) -> Union[list, dict]:
        """
        Interface for planning using RRT algorithm.

        Returns:
            path: A list containing the path waypoints
            path_info: A dictionary containing the path information (success, length, cost, expand)
        """
        # Initialize the tree with the start node
        start_node = Node(self.start, None, 0, 0)
        self.tree[start_node.current] = start_node
        expand = [start_node]

        # Main loop for sampling
        for _ in range(self.sample_num):
            # Generate a random node
            rand_node = self._generate_random_node()
            
            # Find the nearest node in the tree
            nearest_node = self._find_nearest_node(rand_node)
            
            # Expand towards the random node
            new_node = self._expand_node(nearest_node, rand_node)
            
            # Check if the new node is valid (no collision)
            if new_node and new_node.current not in self.tree:
                self.tree[new_node.current] = new_node
                expand.append(new_node)
                
                # Check if we can reach the goal from the new node
                if not self.map_.in_collision(new_node.current, self.goal):
                    # Add goal node to the tree
                    goal_node = Node(self.goal, new_node.current, 
                                    new_node.g + self.get_cost(new_node.current, self.goal), 0)
                    self.tree[goal_node.current] = goal_node
                    expand.append(goal_node)
                    
                    # Extract and return the path
                    path, length, cost = self.extract_path(self.tree)
                    return path, {
                        "success": True,
                        "start": self.start,
                        "goal": self.goal,
                        "length": length,
                        "cost": cost,
                        "expand": expand
                    }

        # If no path found within sample limit
        self.failed_info[1]["expand"] = expand
        return self.failed_info

    def _generate_random_node(self) -> Node:
        """
        Generate a random node in the map, with occasional bias towards the goal.
        
        Returns:
            A random Node object
        """
        # With certain probability, sample the goal point
        if random.random() < self.goal_sample_rate:
            return Node(self.goal, None, 0, 0)
        
        # Otherwise, generate a random point within map bounds
        x_min, x_max = self.map_.bounds[0]
        y_min, y_max = self.map_.bounds[1]
        
        # Generate integer coordinates since we're working with grid maps
        x = random.randint(x_min, x_max)
        y = random.randint(y_min, y_max)
        
        return Node((x, y), None, 0, 0)

    def _find_nearest_node(self, node: Node) -> Node:
        """
        Find the nearest node in the tree to the given node.
        
        Args:
            node: The target node
            
        Returns:
            The nearest node from the tree
        """
        nearest_node = None
        min_dist = float('inf')
        
        for candidate in self.tree.values():
            dist = self.get_cost(candidate.current, node.current)
            if dist < min_dist:
                min_dist = dist
                nearest_node = candidate
                
        return nearest_node

    def _expand_node(self, nearest: Node, target: Node) -> Union[Node, None]:
        """
        Expand from the nearest node towards the target node, up to max_dist.
        
        Args:
            nearest: The nearest node in the tree
            target: The target node to expand towards
            
        Returns:
            A new Node if expansion is possible, None otherwise
        """
        # Calculate the direction from nearest to target
        dx = target.current[0] - nearest.current[0]
        dy = target.current[1] - nearest.current[1]
        distance = math.hypot(dx, dy)
        
        # If distance is less than max_dist, use target directly
        if distance < self.max_dist:
            new_x, new_y = target.current
        else:
            # Otherwise, move max_dist towards target
            ratio = self.max_dist / distance
            new_x = nearest.current[0] + int(round(dx * ratio))
            new_y = nearest.current[1] + int(round(dy * ratio))
        
        new_current = (new_x, new_y)
        
        # Check if the path to new node is collision-free
        if self.map_.in_collision(nearest.current, new_current):
            return None
        
        # Calculate cost to new node
        new_g = nearest.g + self.get_cost(nearest.current, new_current)
        
        return Node(new_current, nearest.current, new_g, 0)
