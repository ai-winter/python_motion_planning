"""
@file: voronoi.py
@author: Wu Maojia
@update: 2025.12.20
"""
import copy
from typing import Union, List, Tuple, Dict, Any
import heapq

import numpy as np
from scipy.spatial import Voronoi

from python_motion_planning.common import Node, TYPES, Grid
from python_motion_planning.path_planner.base_path_planner import BasePathPlanner
from python_motion_planning.path_planner.graph_search.a_star import AStar


class VoronoiPlanner(BasePathPlanner):
    """
    Path planner based on Voronoi diagram.
    Core idea: find the nearest points on the Voronoi diagram to the start and goal,
    plan the path on the Voronoi graph using a base planner, and then concatenate the full path.
    Note that the boundary of grid map passed in Voronoi planner should be filled with obstacles using Grid.fill_boundary_with_obstacles() method. If not, the Voronoi planner will automatically fill.

    Args:
        *args: see the parent class.
        base_planner: base planner class for path planning.
        base_planner_kwargs: keyword arguments for the base planner.
        cover_inflation: determine whether the voronoi candidates cover the inflation region.
        *kwargs: see the parent class.

    Examples:
        >>> map_ = Grid(bounds=[[0, 15], [0, 15]])
        >>> planner = VoronoiPlanner(map_=map_, start=(5, 5), goal=(10, 10))
        >>> path, path_info = planner.plan()
        >>> print(path_info['success'])
        True
        
        >>> planner.map_.type_map[3:10, 6] = TYPES.OBSTACLE
        >>> path, path_info = planner.plan()
        >>> print(path_info['success'])
        True
    """
    def __init__(self, 
            *args, 
            base_planner: BasePathPlanner = AStar, 
            base_planner_kwargs: dict = {},
            cover_inflation: bool = False,
            **kwargs
            ) -> None:
        super().__init__(*args, **kwargs)

        self.base_planner = base_planner
        self.base_planner_kwargs = base_planner_kwargs
        self.base_planner_kwargs["map_"] = self.map_
        self.base_planner_kwargs["start"] = self.start
        self.base_planner_kwargs["goal"] = self.goal
        
        self.cover_inflation = cover_inflation

    def __str__(self) -> str:
        return "Voronoi Planner"

    def find_voronoi_candidates(self, map_: Grid) -> np.ndarray:
        """
        Find Voronoi candidate points for the grid map.

        Args:
            map_: grid map.

        Returns:
            candidates: Voronoi candidate points matrix.
        """
        generators = np.argwhere(map_.data == 1)
        candidates = np.zeros_like(map_.data, dtype=np.bool_)
        if len(generators) < 2:
            return candidates

        vor = Voronoi(generators)

        for ridge in vor.ridge_vertices:
            v1_idx, v2_idx = ridge[:2]
            if v1_idx == -1 or v2_idx == -1:
                continue  # skip infinite ridges
                
            # continuous coordinates
            v1 = vor.vertices[v1_idx]
            v2 = vor.vertices[v2_idx]
            
            v1 = map_.point_float_to_int(v1)
            v2 = map_.point_float_to_int(v2)
        
            line = map_.line_of_sight(v1, v2)

            for point in line:
                candidates[point] = True

        candidates[map_.data == TYPES.OBSTACLE] = False
        if not self.cover_inflation:
            candidates[map_.data == TYPES.INFLATION] = False

        return candidates

    def find_nearest_voronoi_point(self, 
            point: Tuple[float, ...], 
            voronoi_candidates: np.ndarray,
        ) -> Union[Tuple[float, ...], None]:
        """
        Find the nearest Voronoi candidate point to the target point using brute force search.
        
        Args:
            point: target point.
            voronoi_candidates: Voronoi candidate points matrix.

        Returns:
            nearest_point: nearest Voronoi point.
        """
        min_dist = float('inf')
        nearest_point = None

        # Iterate through all Voronoi candidate points to find the nearest one  
        for indices in np.ndindex(voronoi_candidates.shape):
            if voronoi_candidates[indices]:
                candidate_point = indices
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
        voronoi_map = copy.deepcopy(self.map_)
        voronoi_map.fill_boundary_with_obstacles()

        # Compute Voronoi candidate points
        voronoi_candidates = self.find_voronoi_candidates(voronoi_map)

        # If no Voronoi candidates are found, fall back to normal base planner
        if not np.any(voronoi_candidates):
            return self.base_planner(**self.base_planner_kwargs).plan()

        # Find the nearest Voronoi points for start and goal
        start_voronoi = self.find_nearest_voronoi_point(self.start, voronoi_candidates)
        goal_voronoi = self.find_nearest_voronoi_point(self.goal, voronoi_candidates)

        # If no valid Voronoi points found, fall back to normal base planner
        if start_voronoi is None or goal_voronoi is None:
            return self.base_planner(**self.base_planner_kwargs).plan()

        voronoi_map.type_map[voronoi_candidates] = TYPES.FREE
        voronoi_map.type_map[~voronoi_candidates] = TYPES.OBSTACLE

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
            "voronoi_candidates": voronoi_candidates,
            "voronoi_start": start_voronoi,
            "voronoi_goal": goal_voronoi,
            "voronoi_path": voronoi_path
        }

        return final_path, path_info
