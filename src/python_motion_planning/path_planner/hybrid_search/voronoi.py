"""
@file: voronoi.py
@author: Wu Maojia
@update: 2025.10.17
"""
import copy
from typing import Union, List, Tuple, Dict, Any
import heapq

import numpy as np

from python_motion_planning.common import Node, TYPES
from python_motion_planning.path_planner.base_path_planner import BasePathPlanner
from python_motion_planning.path_planner.graph_search.a_star import AStar


class Voronoi(BasePathPlanner):
    """
    Path planner based on Voronoi diagram.
    Core idea: find the nearest points on the Voronoi diagram to the start and goal,
    plan the path on the Voronoi graph using a base planner, and then concatenate the full path.

    Args:
        *args: see the parent class.
        base_planner: base planner class for path planning.
        base_planner_kwargs: keyword arguments for the base planner.
        gradient_threshold: gradient threshold for determining Voronoi candidate points using ESDF map.
        *kwargs: see the parent class.
    """
    def __init__(self, 
            *args, 
            base_planner: BasePathPlanner = AStar, 
            base_planner_kwargs: dict = {},
            gradient_threshold: float = np.sqrt(2)/2-1e-6,
            **kwargs
            ) -> None:
        super().__init__(*args, **kwargs)

        self.base_planner = base_planner
        self.base_planner_kwargs = base_planner_kwargs
        self.base_planner_kwargs["map_"] = self.map_
        self.base_planner_kwargs["start"] = self.start
        self.base_planner_kwargs["goal"] = self.goal
        
        self.gradient_threshold = gradient_threshold  # Gradient threshold for Voronoi candidate points
        self.voronoi_candidates = None  # Voronoi candidate points matrix

    def __str__(self) -> str:
        return "Voronoi"

    @staticmethod
    def find_voronoi_candidates(esdf, threshold):
        """
        Find Voronoi candidate points using gradients of ESDF map. This is an approximation method.

        Args:
            esdf: ESDF map.
            threshold: gradient threshold.

        Returns:
            candidates: Voronoi candidate points matrix.
        """
        grad_x = np.gradient(esdf, axis=0)
        grad_y = np.gradient(esdf, axis=1)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        candidates = grad_magnitude < threshold
        free_space = esdf > 0
        candidates = candidates & free_space
        return candidates

    def find_nearest_voronoi_point(self, point: Tuple[float, ...]) -> Union[Tuple[float, ...], None]:
        """
        Find the nearest Voronoi candidate point to the target point
        
        Args:
            point: target point.

        Returns:
            nearest_point: nearest Voronoi point.
        """
        if self.voronoi_candidates is None or not np.any(self.voronoi_candidates):
            return None

        min_dist = float('inf')
        nearest_point = None
        # Iterate through all Voronoi candidate points to find the nearest one
        for i in range(self.voronoi_candidates.shape[0]):
            for j in range(self.voronoi_candidates.shape[1]):
                if self.voronoi_candidates[i, j]:
                    candidate_point = (i, j)
                    dist = self.map_.get_distance(point, candidate_point)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_point = candidate_point

        return nearest_point

    def plan(self) -> Union[List[Tuple[float, ...]], Dict[str, Any]]:
        """
        Execute the path planning:
        1. Compute Voronoi candidate points
        2. Find the nearest Voronoi points for start and goal
        3. Plan the path on the Voronoi graph
        4. Concatenate the full path (start -> Voronoi start -> ... -> Voronoi goal -> goal)
        
        Returns:
            path: A list containing the path waypoints
            path_info: A dictionary containing the path information
        """
        # Compute Voronoi candidate points
        self.voronoi_candidates = self.find_voronoi_candidates(
            self.map_.esdf, 
            threshold=self.gradient_threshold
        )
        
        # If no Voronoi candidates are found, fall back to normal base planner
        if not np.any(self.voronoi_candidates):
            return self.base_planner(**self.base_planner_kwargs).plan()

        # Find the nearest Voronoi points for start and goal
        start_voronoi = self.find_nearest_voronoi_point(self.start)
        goal_voronoi = self.find_nearest_voronoi_point(self.goal)
        
        # If no valid Voronoi points found, fall back to normal base planner
        if start_voronoi is None or goal_voronoi is None:
            return self.base_planner(**self.base_planner_kwargs).plan()

        voronoi_map = copy.deepcopy(self.map_)
        voronoi_map.type_map[self.voronoi_candidates] = TYPES.FREE
        voronoi_map.type_map[~self.voronoi_candidates] = TYPES.OBSTACLE

        self.base_planner_kwargs["map_"] = voronoi_map
        self.base_planner_kwargs["start"] = start_voronoi
        self.base_planner_kwargs["goal"] = goal_voronoi

        voronoi_path, voronoi_path_info = self.base_planner(**self.base_planner_kwargs).plan()
    
        # If Voronoi path planning fails, fall back to normal base planner
        if not voronoi_path_info["success"]:
            self.base_planner_kwargs["map_"] = self.map_
            self.base_planner_kwargs["start"] = self.start
            self.base_planner_kwargs["goal"] = self.goal
            return self.base_planner(**self.base_planner_kwargs).plan()

        # Compute total path length and cost
        start_segment_len = self.map_.get_distance(self.start, start_voronoi)
        end_segment_len = self.map_.get_distance(goal_voronoi, self.goal)
        total_length = voronoi_path_info["length"] + start_segment_len + end_segment_len

        start_segment_cost = self.get_cost(self.start, start_voronoi)
        end_segment_cost = self.get_cost(goal_voronoi, self.goal)
        total_cost = voronoi_path_info["cost"] + start_segment_cost + end_segment_cost

        # Concatenate the final path
        final_path = [self.start] + voronoi_path + [self.goal]

        # Collect path information
        path_info = {
            "success": True,
            "start": self.start,
            "goal": self.goal,
            "length": total_length,
            "cost": total_cost,
            "expand": voronoi_path_info["expand"],
            "voronoi_candidates": self.voronoi_candidates,
            "voronoi_start": start_voronoi,
            "voronoi_goal": goal_voronoi,
            "voronoi_path": voronoi_path
        }

        return final_path, path_info
