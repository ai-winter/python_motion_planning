"""
@file: base_visualizer.py
@author: Wu Maojia
@update: 2025.12.20
"""
from typing import Union, Dict, List, Tuple, Any
from abc import ABC, abstractmethod

import numpy as np

from python_motion_planning.common.utils import Geometry

class BaseVisualizer(ABC):
    """
    Base visualizer for motion planning.
    """
    def __init__(self):
        self.dim = None
        self.trajs = {}

    def __del__(self):
        self.close()

    @abstractmethod
    def show(self):
        """
        Show plot.
        """
        pass
    
    @abstractmethod
    def close(self):
        """
        Close plot.
        """
        pass

    def get_traj_info(self, 
            rid: int, 
            ref_path: List[Tuple[float, ...]], 
            goal_pose: np.ndarray, 
            goal_dist_tol: float, 
            goal_orient_tol: float
        ) -> Dict[str, Any]:
        """
        Get trajectory information.

        Args:
            rid: Robot ID.
            ref_path: Reference pose path (world frame).
            goal_pose: Goal pose.
            goal_dist_tol: Distance tolerance for goal.
            goal_orient_tol: Orientation tolerance for goal.
        """
        traj = self.trajs[rid]

        info = {
            "traj_length": 0.0,
            "navigation_error": None,
            "DTW": None,
            "nDTW": None,
            "success": False,
            "dist_success": False, 
            "oracle_success": False,
            "oracle_dist_success": False,
            "success_time": None,
            "dist_success_time": None,
            "oracle_success_time": None,
            "oracle_dist_success_time": None,
        }

        goal_pos = goal_pose[:self.dim]
        goal_orient = goal_pose[self.dim:]

        for i in range(len(traj["poses"])):
            pose = traj["poses"][i]
            time = traj["time"][i]
            
            pos = pose[:self.dim]
            orient = pose[self.dim:]

            if i > 0:
                info["traj_length"] += np.linalg.norm(pos - traj["poses"][i-1][:self.dim])

            if np.linalg.norm(pos - goal_pos) <= goal_dist_tol:
                if not info["oracle_dist_success"]:
                    info["oracle_dist_success"] = True
                    info["oracle_dist_success_time"] = time

                if not info["dist_success"]:
                    info["dist_success"] = True
                    info["dist_success_time"] = time

                if np.abs(Geometry.regularize_orient(orient - goal_orient)) <= goal_orient_tol:
                    if not info["oracle_success"]:
                        info["oracle_success"] = True
                        info["oracle_success_time"] = time  

                    if not info["success"]:
                        info["success"] = True
                        info["success_time"] = time
                    
                else:
                    info["success"] = False
                    info["success_time"] = None
                
            else:
                info["success"] = False
                info["success_time"] = None
                info["dist_success"] = False
                info["dist_success_time"] = None

        info["navigation_error"] = float(np.linalg.norm(traj["poses"][-1][:self.dim] - goal_pos))
        info["traj_length"] = float(info["traj_length"])
        info["DTW"], info["nDTW"] = self.calc_dtw_ndtw(np.array(traj["poses"])[:, :self.dim], np.array(ref_path)[:, :self.dim])
        return info

    def calc_dtw_ndtw(self, path1: np.ndarray, path2: np.ndarray) -> Tuple[float, float]:
        """
        Compute the Dynamic Time Warping (DTW) and normalized DTW (nDTW)
        between two N-dimensional paths.

        Args:
            path1 (np.ndarray): Path 1, shape (N, D)
            path2 (np.ndarray): Path 2, shape (M, D)

        Returns:
            dtw: accumulated dynamic time warping distance
            ndtw: normalized DTW in [0, 1], higher means more similar
        
        Reference:
            [1] General Evaluation for Instruction Conditioned Navigation using Dynamic Time Warping
        """
        # Input validation
        if path1.ndim != 2 or path2.ndim != 2:
            raise ValueError("Both paths must be 2D arrays with shape (T, D).")
        if path1.shape[1] != path2.shape[1]:
            raise ValueError("Paths must have the same dimensionality.")

        N, M = len(path1), len(path2)

        # Initialize DTW cost matrix
        dtw_matrix = np.full((N + 1, M + 1), np.inf)
        dtw_matrix[0, 0] = 0.0

        # Fill the DTW matrix
        for i in range(1, N + 1):
            for j in range(1, M + 1):
                # Euclidean distance between points
                cost = np.linalg.norm(path1[i - 1] - path2[j - 1])
                # Accumulate cost with the best previous step
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],      # insertion
                    dtw_matrix[i, j - 1],      # deletion
                    dtw_matrix[i - 1, j - 1]   # match
                )

        # Final DTW distance
        dtw = dtw_matrix[N, M]

        # Normalized DTW: exponential decay with respect to path length
        max_len = max(N, M)
        ndtw = np.exp(-dtw / (max_len + 1e-8))  # nDTW ∈ (0, 1]

        return float(dtw), float(ndtw)
