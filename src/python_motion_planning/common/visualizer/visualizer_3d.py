"""
@file: visualizer_3d.py
@author: Wu Maojia
@update: 2025.12.20
"""
from typing import Union, Dict, List, Tuple, Any
from collections import namedtuple
import time
import os

import numpy as np
import pyvista as pv

from python_motion_planning.common.visualizer.base_visualizer import BaseVisualizer
from python_motion_planning.controller import BaseController
from python_motion_planning.common.env import TYPES, ToySimulator, Grid, CircularRobot, Node
from python_motion_planning.common.utils import Geometry

class Visualizer3D(BaseVisualizer):
    """
    Simple 3D visualizer for motion planning using pyvista.

    Args:
        window_size: Window size (width, height) (pyvista window size, unit: pixel).
        off_screen: `off_screen` argument for pyvista. Renders off screen when True. Useful for automated screenshots.
        show_axes: Whether to show axes for pyvista.
        cmap_dict: Color map for 3d voxel visualization.
    """
    def __init__(self,  
                window_size: tuple = (1200, 900),
                off_screen: bool = False,
                show_axes: bool = True,
                cmap_dict: dict = {
                    TYPES.FREE: "#ffffff",
                    TYPES.OBSTACLE: "#000000",
                    TYPES.START: "#ff0000",
                    TYPES.GOAL: "#1155cc",
                    TYPES.INFLATION: "#ffccff",
                    TYPES.EXPAND: "#eeeeee",
                    TYPES.CUSTOM: "#bbbbbb",
                }
            ):
        super().__init__()
        self.pv_plotter = pv.Plotter(window_size=list(window_size), off_screen=off_screen)
        if show_axes: 
            self.pv_plotter.show_axes()
        self.pv_actors = {} 

        # colors
        self.cmap_dict = cmap_dict

    def plot_grid_map(self, grid_map: Grid, equal: bool = False, alpha_3d: dict = {
                            TYPES.FREE: 0.0,
                            TYPES.OBSTACLE: 0.1,
                            TYPES.START: 0.5,
                            TYPES.GOAL: 0.5,
                            TYPES.INFLATION: 0.0,
                            TYPES.EXPAND: 0.01,
                            TYPES.CUSTOM: 0.1,
                        }) -> None:
        '''
        Plot grid map with static obstacles.

        Args:
            map: Grid map or its type map.
            equal: Whether to set axis equal.
            alpha_3d: Alpha of occupancy for 3d visualization.
        '''
        if grid_map.dim != 3:
            raise ValueError(f"Grid map dimension must be 3.")
        
        self.grid_map = grid_map
        self.dim = grid_map.dim
        type_data = grid_map.type_map.data

        nx, ny, nz = type_data.shape

        for key, color in self.cmap_dict.items():
            alpha = alpha_3d.get(key, 0.0)
            if alpha < 1e-6:
                continue

            mask = (type_data == key)
            if not np.any(mask):
                continue

            # voxels
            points = np.argwhere(mask)

            # map → world
            points = np.array([
                self.grid_map.map_to_world(p)
                for p in points
            ])

            cloud = pv.PolyData(points)
            glyph = cloud.glyph(
                geom=pv.Cube(),
                scale=False,
                factor=self.grid_map.resolution
            )

            actor = self.pv_plotter.add_mesh(
                glyph,
                color=color,
                opacity=alpha,
                show_edges=False
            )

            self.pv_actors[f"voxels_{key}"] = actor

    def plot_expand_tree(self, expand_tree: Dict[Union[Tuple[int, ...], Tuple[float, ...]], Node], 
                        edge_color: str = "#e377c2", 
                        linewidth: float = 1.0, 
                        node_alpha: float = 1.0,
                        edge_alpha: float = 1.0,
                        map_frame: bool = True) -> None:
        """
        Visualize an expand tree (e.g. RRT).
        
        Args:
            expand_tree: Dict mapping coordinate tuple -> Node (world frame).
            edge_color: Color of the edges (parent -> child).
            linewidth: Line width of edges.
            map_frame: whether path is in map frame or not (world frame)
        """
        if not isinstance(expand_tree, list):
            expand_tree = [expand_tree]

        points = []
        lines = []

        idx = 0
        for tree in expand_tree:
            for _, node in tree.items():
                cur = node.current
                if map_frame:
                    cur = self.grid_map.map_to_world(cur)

                points.append(cur)

                if node.parent is not None:
                    parent = node.parent
                    if map_frame:
                        parent = self.grid_map.map_to_world(parent)

                    points.append(parent)
                    lines.append([idx, idx + 1])
                    idx += 2
                else:
                    idx += 1

        if not points:
            return

        points = np.array(points)
        poly = pv.PolyData(points)

        if lines:
            cells = np.hstack([[2, l[0], l[1]] for l in lines])
            poly.lines = cells

        self.pv_plotter.add_mesh(
            poly,
            color=edge_color,
            line_width=linewidth
        )


    def plot_path(self, path: List[Union[Tuple[int, ...], Tuple[float, ...]]], 
                    color: str = "#13ae00", 
                    linewidth: float = 5, map_frame: bool = True) -> None:
        '''
        Plot path-like information.
        The meaning of parameters are similar to pyvista.Plotter.add_mesh (https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.add_mesh#pyvista.Plotter.add_mesh).

        Args:
            path: point list of path
            color: color of path
            linewidth: linewidth of path
            map_frame: whether path is in map frame or not (world frame)
        '''
        if len(path) == 0:
            return

        if map_frame:
            path = [self.grid_map.map_to_world(point) for point in path]

        path = np.array(path)

        path_line = pv.lines_from_points(path)
        self.pv_plotter.add_mesh(
            path_line,
            color=color,
            line_width=linewidth
        )

    def set_title(self, title: str) -> None:
        """
        Set title.

        Args:
            title: Title.
        """
        self.pv_plotter.add_text(title, position='upper_edge', font_size=14, color='black')

    def clean(self):
        """
        Clean plot.
        """
        self.pv_plotter.clear()
        self.pv_actors = {}

    def update(self):
        """
        Update plot.
        """
        self.pv_plotter.render()

    def savefig(self, filename, *args, **kwargs):
        """
        Save figure. 

        Args:
            filename: Filename to save.
            *args: See pyvista.Plotter.screenshot.
            **kwargs: See pyvista.Plotter.screenshot.
        """
        self.pv_plotter.screenshot(filename=filename, *args, **kwargs)

    def show(self):
        """
        Show plot.
        """
        self.pv_plotter.reset_camera()
        self.pv_plotter.show(interactive=True, auto_close=False)

    def close(self):
        """
        Close plot.
        """
        self.pv_plotter.close()
