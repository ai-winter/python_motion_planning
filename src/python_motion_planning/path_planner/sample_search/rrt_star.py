"""
@file: rrt_star.py
@author: Wu Maojia
@update: 2025.12.19
"""
import math
import random
from typing import Union, Dict, List, Tuple, Any

import numpy as np
import faiss

from python_motion_planning.common import Node, Grid, TYPES
from python_motion_planning.path_planner.sample_search import RRT
from python_motion_planning.common.utils.child_tree import ChildTree


class RRTStar(RRT):
    """
    Class for RRT* (Rapidly-exploring Random Tree Star) path planner.

    RRT* extends RRT by:
        1. Selecting the best parent (minimum cost) for a new node.
        2. Rewiring nearby nodes through the new node if it improves their cost.

    Args:
        *args: see parent class.
        rewire_radius: Neighborhood radius for rewiring (If None, adaptively calculated with gamma factor).
        gamma: A factor for calculating rewire_radius. For details, see [1]. (Disabled when rewire_radius is not None)
        stop_until_sample_num: Stop until sample number limitation is reached, otherwise stop when goal is found.
        propagate_cost_to_children: Whether to propagate cost to children. This is a fix for ensuring the correctness of g-value of each node. But it may slow the algorithm a little.
        *kwargs: see parent class.

    References:
        [1] Sampling-based Algorithms for Optimal Motion Planning

    Examples:
        >>> map_ = Grid(bounds=[[0, 15], [0, 15]])
        >>> planner = RRTStar(map_=map_, start=(5, 5), goal=(10, 10))
        >>> path, path_info = planner.plan()
        >>> print(path_info['success'])
        True
        
        >>> planner = RRTStar(map_=map_, start=(5, 5), goal=(10, 10), sample_num=1000, stop_until_sample_num=True)
        >>> path, path_info = planner.plan()
        >>> print(path_info['success'])
        True
    """

    def __init__(self, *args,
                 rewire_radius: float = None,
                 gamma: float = 50.0,
                 stop_until_sample_num: int = False,
                 propagate_cost_to_children: bool = True,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rewire_radius = rewire_radius
        self.gamma = gamma
        self.stop_until_sample_num = stop_until_sample_num
        self.best_results = self.failed_info
        self.propagate_cost_to_children = propagate_cost_to_children

        self._tree = None
        self._child = None

    def __str__(self) -> str:
        return "RRT*"

    def plan(self) -> Union[List[Tuple[float, ...]], Dict[str, Any]]:
        """
        RRT* path planning algorithm implementation.

        Returns:
            path: A list containing the path waypoints
            path_info: A dictionary containing path information
        """
        # Initialize tree with start node
        self._tree = {}
        self._child = ChildTree()
        start_node = Node(self.start, None, 0, 0)
        self._tree[self.start] = start_node

        # Initialize FAISS index
        if self.use_faiss:
            faiss_index = faiss.IndexFlatL2(self.dim)
            faiss_nodes = []
            self._faiss_add_node(start_node, faiss_index, faiss_nodes)

        for i in range(self.sample_num):
            # Generate random sample
            node_rand = self._generate_random_node()

            if node_rand.current in self._tree:
                continue

            # Find nearest node
            node_nearest = self._get_nearest_node(self._tree, node_rand, faiss_index, faiss_nodes)

            # Steer towards random sample
            node_new = self._steer(node_nearest, node_rand)
            if node_new is None:
                continue

            # Collision check
            if self.map_.in_collision(
                self.map_.point_float_to_int(node_nearest.current), 
                self.map_.point_float_to_int(node_new.current)
                ):
                continue

            # Find nearby nodes for choosing best parent
            near_nodes = self._get_near_nodes(node_new, faiss_index, faiss_nodes)

            # Choose parent with minimum cost
            min_parent = node_nearest
            min_cost = node_nearest.g + self.get_cost(node_nearest.current, node_new.current)

            for node in near_nodes:
                if self.map_.in_collision(
                    self.map_.point_float_to_int(node.current), 
                    self.map_.point_float_to_int(node_new.current)
                    ):
                    continue
                cost = node.g + self.get_cost(node.current, node_new.current)
                if cost < min_cost:
                    min_parent = node
                    min_cost = cost

            # Add new node
            if self.propagate_cost_to_children:
                self._child.remove(node_new.parent, node_new.current)
                self._child.add(min_parent.current, node_new.current)

            node_new.parent = min_parent.current
            node_new.g = min_cost
            self._tree[node_new.current] = node_new
            if self.use_faiss:
                self._faiss_add_node(node_new, faiss_index, faiss_nodes)

            # Rewire nearby nodes through new node
            for node in near_nodes:
                if node.current == min_parent.current:
                    continue
                if self.map_.in_collision(
                    self.map_.point_float_to_int(node.current),
                    self.map_.point_float_to_int(node_new.current)
                    ):
                    continue

                new_cost = node_new.g + self.get_cost(node_new.current, node.current)
                if new_cost < node.g:
                    if self.propagate_cost_to_children:
                        self._child.remove(node.parent, node.current)
                        self._child.add(node_new.current, node.current)

                    node.parent = node_new.current
                    node.g = new_cost
                    self._tree[node.current] = node

                    if self.propagate_cost_to_children:
                        self._propagate_cost_to_children(node)

            # Check goal connection
            dist_to_goal = self.get_cost(node_new.current, self.goal)
            if dist_to_goal <= self.max_dist:
                if not self.map_.in_collision(
                    self.map_.point_float_to_int(node_new.current), 
                    self.map_.point_float_to_int(self.goal)
                    ):
                    goal_cost = node_new.g + dist_to_goal
                    if self.goal not in self._tree or goal_cost < self._tree[self.goal].g:
                        if node_new.current == self.goal:
                            self._tree[self.goal] = node_new
                        else:
                            self._tree[self.goal] = Node(
                                self.goal,
                                node_new.current,
                                goal_cost,
                                0
                            )
                        path, length, cost = self.extract_path(self._tree)
                        self.best_results = path, {
                            "success": True,
                            "start": self.start,
                            "goal": self.goal,
                            "length": length,
                            "cost": cost,
                            "expand": self._tree,
                        }

                        if not self.stop_until_sample_num:
                            return self.best_results

        n = len(self._tree) + 1
        radius = self.gamma * ((math.log(n) / n) ** (1 / self.dim))

        # Planning stopped
        self.best_results[1]["expand"] = self._tree
        return self.best_results

    def _get_near_nodes(self, node_new: Node, index=None, nodes=None) -> List[Node]:
        """
        Get nearby nodes within rewiring radius.

        Args:
            node_new: Newly added node
            index: FAISS index (required when `use_faiss`=True)
            nodes: List of nodes in FAISS index (required when `use_faiss`=True)

        Returns:
            near_nodes: List of nearby nodes
        """
        # Adaptive radius from RRT* theory
        if self.rewire_radius is None:
            n = len(self._tree) + 1
            radius = self.gamma * ((math.log(n) / n) ** (1 / self.dim))
            if self.discrete:
                radius = max(1, radius)
        else:
            radius = self.rewire_radius

        near_nodes = []
        
        if self.use_faiss:
            # range search using faiss
            query = np.asarray(node_new.current, dtype=np.float32).reshape(1, -1)
            lims, D, I = index.range_search(query, radius * radius)
            for idx in I:
                near_nodes.append(nodes[idx])

        else:
            # brute force radius search
            for node in self._tree.values():
                if self.get_cost(node.current, node_new.current) <= radius:
                    near_nodes.append(node)

        return near_nodes

    def _propagate_cost_to_children(self, node: Node):
        """
        Propagate cost update to children of a node.

        Args:
            node: Node to propagate cost to children
        """
        child_set = self._child[node.current]
        if child_set is None:
            return

        for child in child_set:
            node_child = self._tree.get(child)

            if node_child is not None:
                old_g = node_child.g
                node_child.g = node.g + self.get_cost(
                    node.current, node_child.current
                )
                if not math.isclose(node_child.g, old_g):
                    self._propagate_cost_to_children(node_child)
