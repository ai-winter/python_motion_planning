"""
@file: jps.py
@author: Wu Maojia, Yang Haodong
@update: 2025.10.6
"""
from typing import Union, List, Tuple, Dict, Any
import heapq
from itertools import product

import numpy as np

from python_motion_planning.common import BaseMap, Grid, Node, TYPES
from python_motion_planning.path_planner.graph_search.a_star import AStar

class JPS(AStar):
    """
    Class for Jump Point Search (JPS) path planner.

    JPS is an optimization of the A* algorithm that identifies jump points to reduce the number of nodes 
    that need to be explored, significantly improving pathfinding efficiency on grid maps.

    This implementation does not support non-diagonal option.

    Args:
        *args: see the parent class.
        **kwargs: see the parent class.

    References:
        [1] Online Graph Pruning for Pathfinding On Grid Maps

    Examples:
        >>> map_ = Grid(bounds=[[0, 15], [0, 15]])
        >>> planner = JPS(map_=map_, start=(5, 5), goal=(10, 10))
        >>> planner.plan()
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Precompute all possible movement directions for N-dimensional grid
        self.directions = self.map_._diagonal_offsets if self.diagonal else self.map_._orthogonal_offsets

    def __str__(self) -> str:
        return "Jump Point Search (JPS)"

    def plan(self) -> Union[List[Tuple[float, ...]], Dict[str, Any]]:
        """
        Interface for planning using Jump Point Search.

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

            # Skip if node is already processed
            if node.current in CLOSED:
                continue

            # Goal found
            if node.current == self.goal:
                CLOSED[node.current] = node
                path, length, cost = self.extract_path(CLOSED)
                return path, {
                    "success": True, 
                    "start": self.start, 
                    "goal": self.goal, 
                    "length": length, 
                    "cost": cost, 
                    "expand": CLOSED
                }

            # Find all jump points from current node
            jump_points = []
            for direction in self.directions:
                jp = self.jump(node, direction)
                if jp and jp.current not in CLOSED:
                    jp.parent = node.current
                    jp.g = node.g + self.get_cost(node.current, jp.current)
                    jp.h = self.get_heuristic(jp.current)
                    jump_points.append(jp)

            # Add jump points to OPEN list
            for jp in jump_points:
                heapq.heappush(OPEN, jp)
                
                # Check if we found the goal
                if jp.current == self.goal:
                    # Push goal node and break
                    heapq.heappush(OPEN, jp)
                    break

            # Add current node to CLOSED list
            CLOSED[node.current] = node

        # If no path found
        self.failed_info[1]["expand"] = CLOSED
        return self.failed_info

    def jump(self, node: Node, direction: Node) -> Union[Node, None]:
        """
        Recursively search for jump points in a given direction.

        Args:
            node: Current node to start the jump from
            direction: Direction of the jump (as a Node with integer coordinates)

        Returns:
            jump_point: The found jump point or None if no jump point exists
        """
        # Calculate next node in the given direction
        next_coords = tuple(n + d for n, d in zip(node.current, direction.current))
        next_node = Node(next_coords, node.current, 0, 0)

        # Check if next node is valid (within bounds and not an obstacle)
        if not self.map_.is_expandable(next_node.current, node.current):
            return None

        # Check if we've reached the goal
        if next_node.current == self.goal:
            return next_node

        # Check for forced neighbors
        if self._has_forced_neighbors(next_node, direction):
            return next_node

        # For diagonal directions, check if we can jump in orthogonal directions
        if all(d != 0 for d in direction.current):
            # Check each orthogonal component of the diagonal direction
            for i in range(self.dim):
                if direction.current[i] != 0:
                    # Create orthogonal direction vector
                    ortho_dir = [0] * self.dim
                    ortho_dir[i] = direction.current[i]
                    ortho_dir = Node(tuple(ortho_dir))
                    
                    # If there's a jump point in this orthogonal direction,
                    # current node is a jump point
                    if self.jump(next_node, ortho_dir):
                        return next_node

        # Continue jumping in the same direction
        return self.jump(next_node, direction)

    def _has_forced_neighbors(self, node: Node, direction: Node) -> bool:
        """
        Check if the current node has any forced neighbors that would make it a jump point.

        Args:
            node: Current node to check
            direction: Direction of travel to reach this node

        Returns:
            bool: True if there are forced neighbors, False otherwise
        """
        node_coords = node.current
        dir_coords = direction.current

        # Check all possible directions perpendicular to the current direction
        for i in range(self.dim):
            if dir_coords[i] == 0:
                continue  # Skip dimensions not part of current direction
                
            # Check both positive and negative directions in perpendicular dimensions
            for j in range(self.dim):
                if i == j or dir_coords[j] != 0:
                    continue  # Skip same dimension or already part of direction
                    
                # Check both directions in the perpendicular dimension
                for sign in [-1, 1]:
                    # Calculate neighbor coordinates
                    neighbor = list(node_coords)
                    neighbor[j] += sign
                    neighbor = tuple(neighbor)
                    
                    # Calculate opposite neighbor (obstacle check)
                    opposite = list(neighbor)
                    opposite[i] -= dir_coords[i]
                    opposite = tuple(opposite)
                    
                    # If neighbor is valid and opposite is an obstacle,
                    # we have a forced neighbor
                    if (self.map_.is_expandable(neighbor, node.current) and 
                        not self.map_.is_expandable(opposite, node.current)):
                        return True
                        
        return False
