"""
@file: rrt_connect.py
@author: Wu Maojia
@update: 2025.10.10
"""
from typing import Union, Dict, List, Tuple, Any

from python_motion_planning.common import Node
from python_motion_planning.path_planner.sample_search.rrt import RRT

class RRTConnect(RRT):
    """
    Class for RRT-Connect path planner, an improved version of RRT.

    RRT-Connect uses two trees (one from start, one from goal) that grow towards
    each other, which typically results in faster convergence than standard RRT.

    Args:
        *args: see the parent class.
        *kwargs: see the parent class.

    References:
        [1] RRT-connect: An efficient approach to single-query path planning.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Two trees for bidirectional search: one from start, one from goal
        self.tree_a = None  # Tree originating from start point
        self.tree_b = None  # Tree originating from goal point

    def __str__(self) -> str:
        return "Rapidly-exploring Random Tree Connect (RRT-Connect)"

    def plan(self) -> Union[List[Tuple[float, ...]], Dict[str, Any]]:
        """
        RRT-Connect path planning algorithm implementation.

        Returns:
            path: A list containing the path waypoints
            path_info: A dictionary containing path information
        """
        # Initialize both trees with start and goal nodes respectively
        self.tree_a = {self.start: Node(self.start, None, 0, 0)}
        self.tree_b = {self.goal: Node(self.goal, None, 0, 0)}

        # Main planning loop
        for _ in range(self.sample_num):
            # Generate random sample node
            node_rand = self._generate_random_node()
            
            # Extend tree A towards random sample
            node_new_a = self._extend(self.tree_a, node_rand)
            if node_new_a:
                # Try to connect tree B to the new node from tree A
                if self._connect(self.tree_b, node_new_a.current):
                    tree_a_goal = self.tree_a.get(self.goal)
                    if tree_a_goal is not None and tree_a_goal.parent is None:
                        self.tree_a, self.tree_b = self.tree_b, self.tree_a
                    # Path found - combine paths from both trees
                    path_a, length_a, cost_a = self._extract_subpath(self.tree_a, node_new_a.current, self.start)
                    path_b, length_b, cost_b = self._extract_subpath(self.tree_b, node_new_a.current, self.goal)
                    path_a = path_a[::-1]
                    # Combine paths (remove duplicate meeting point)
                    full_path = path_a + path_b[1:]
                    total_length = length_a + length_b
                    total_cost = cost_a + cost_b
                    
                    return full_path, {
                        "success": True,
                        "start": self.start,
                        "goal": self.goal,
                        "length": total_length,
                        "cost": total_cost,
                        "expand": [self.tree_a, self.tree_b],
                    }
            
            # Swap trees to maintain bidirectional growth
            self.tree_a, self.tree_b = self.tree_b, self.tree_a

        # If loop exits without return, planning failed
        self.failed_info[1]["expand"] = [self.tree_a, self.tree_b]
        return self.failed_info

    def _extend(self, tree: Dict[Tuple[int, ...], Node], node_rand: Node) -> Union[Node, None]:
        """
        Extend the tree towards a random node, adding at most one new node.
        
        Args:
            tree: The tree to extend
            node_rand: The target node to extend towards
            
        Returns:
            The new node added to the tree, or None if no node was added
        """
        # Find nearest node in the tree
        node_near = self._get_nearest_node(tree, node_rand)
        
        # Steer towards the random node
        node_new = self._steer(node_near, node_rand)
        if node_new is None:
            return None
            
        # Check if path is collision-free
        if not self.map_.in_collision(node_near.current, node_new.current):
            # Add new node to the tree
            node_new.parent = node_near.current
            node_new.g = node_near.g + self.get_cost(node_near.current, node_new.current)
            tree[node_new.current] = node_new
            return node_new
            
        return None

    def _connect(self, tree: Dict[Tuple[int, ...], Node], target: Tuple) -> bool:
        """
        Connect the tree to a target point by repeatedly extending towards it.
        
        Args:
            tree: The tree to connect
            target: The target point to connect to
            
        Returns:
            True if connection successful, False otherwise
        """
        while True:
            # Create node for the target
            node_target = Node(target, None, 0, 0)
            
            # Find nearest node in the tree
            node_near = self._get_nearest_node(tree, node_target)
            
            # Check distance to target
            dist = self.get_cost(node_near.current, target)
            
            # If close enough, check final connection
            if dist <= self.max_dist:
                if not self.map_.in_collision(node_near.current, target):
                    # Add target to tree
                    node_new = Node(target, node_near.current, 
                                   node_near.g + dist, 0)
                    tree[target] = node_new
                    return True
                return False
                
            # Otherwise, extend towards target
            node_new = self._steer(node_near, node_target)
            if node_new is None:
                return False
                
            # Check collision
            if self.map_.in_collision(node_near.current, node_new.current):
                return False
                
            # Add new node to tree
            node_new.parent = node_near.current
            node_new.g = node_near.g + self.get_cost(node_near.current, node_new.current)
            tree[node_new.current] = node_new

    def _extract_subpath(self, tree: dict, end_point: tuple, root: tuple) -> Tuple[List[Tuple[float, ...]], float, float]:
        """
        Extract a subpath from the root of the tree to the end_point.

        Args:
            tree: Tree to extract path from
            end_point: End point of the subpath
            root: Root point of the tree

        Returns:
            path: the subpath
            length: length of the subpath
            cost: cost of the subpath
        """
        length = 0
        cost = 0
        node = tree.get(end_point)
        path = [node.current]
        
        while node.current != root:
            node_parent = tree.get(node.parent)
            length += self.map_.get_distance(node.current, node_parent.current)
            cost += self.get_cost(node.current, node_parent.current)
            node = node_parent
            path.append(node.current)
            
        return path, length, cost
