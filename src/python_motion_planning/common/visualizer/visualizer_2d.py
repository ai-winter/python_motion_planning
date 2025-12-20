"""
@file: visualizer_2d.py
@author: Wu Maojia, Yang Haodong 
@update: 2025.12.20
"""
from typing import Union, Dict, List, Tuple, Any
from collections import namedtuple
import time
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import animation
import matplotlib.patheffects as path_effects

from python_motion_planning.common.visualizer.base_visualizer import BaseVisualizer
from python_motion_planning.controller import BaseController
from python_motion_planning.common.env import TYPES, ToySimulator, Grid, CircularRobot, Node
from python_motion_planning.common.utils import Geometry

class Visualizer2D(BaseVisualizer):
    """
    Simple visualizer for motion planning using matplotlib.

    Args:
        figname: Figure name (window title).
        figsize: Figure size (width, height) (matplotlib figure size, unit: inch).
        cmap_dict: Color map for 2d visualization.
        zorder: Zorder for 2d matplotlib visualization.
    """
    def __init__(self, 
                figname: str = "", 
                figsize: tuple = (10, 8), 
                cmap_dict: dict = {
                    TYPES.FREE: "#ffffff",
                    TYPES.OBSTACLE: "#000000",
                    TYPES.START: "#ff0000",
                    TYPES.GOAL: "#1155cc",
                    TYPES.INFLATION: "#ffccff",
                    TYPES.EXPAND: "#eeeeee",
                    TYPES.CUSTOM: "#bbbbbb",
                },
                zorder: dict = {
                    'grid_map': 10,
                    'voxels': 10,
                    'esdf': 20,
                    'expand_tree_edge': 30,
                    'expand_tree_node': 40,
                    'path_2d': 50,
                    'path_3d': 700,
                    'traj': 60,
                    'lookahead_pose_node': 70,
                    'lookahead_pose_orient': 80,
                    'pred_traj': 90,
                    'robot_circle': 100,
                    'robot_orient': 110,
                    'robot_text': 120,
                    'env_info_text': 10000
                }
            ):
        self.fig = plt.figure(figname, figsize=figsize)
        self.ax = self.fig.add_subplot()
        self.ani = None

        # colors
        self.cmap_dict = cmap_dict
        # self.norm = mcolors.BoundaryNorm(list(range(len(self.cmap_dict))), len(self.cmap_dict))

        self.zorder = zorder

        self.cmap = mcolors.ListedColormap([info for info in self.cmap_dict.values()])
        self.norm = mcolors.BoundaryNorm([i for i in range(self.cmap.N + 1)], self.cmap.N)
        self.grid_map = None
        self.dim = None

        self.trajs = {}

    def __del__(self):
        self.close()

    def plot_grid_map(self, grid_map: Grid, equal: bool = False,
                        show_esdf: bool = False, alpha_esdf: float = 0.5) -> None:
        '''
        Plot grid map with static obstacles.

        Args:
            map: Grid map or its type map.
            equal: Whether to set axis equal.
            show_esdf: Whether to show esdf.
            alpha_esdf: Alpha of esdf.
        '''
        if grid_map.dim != 2:
            raise ValueError(f"Grid map dimension must be 2.")

        self.grid_map = grid_map
        self.dim = grid_map.dim
        type_data = grid_map.type_map.data

        plt.imshow(
            np.transpose(type_data), 
            cmap=self.cmap, 
            norm=self.norm, 
            origin='lower', 
            interpolation='nearest', 
            extent=[*grid_map.bounds[0], *grid_map.bounds[1]],
            zorder=self.zorder['grid_map'],
            )

        if show_esdf:   # draw esdf hotmap
            plt.imshow(
                np.transpose(grid_map.esdf),
                cmap="jet",
                origin="lower",
                interpolation="nearest",
                extent=[*grid_map.bounds[0], *grid_map.bounds[1]],
                alpha=alpha_esdf,
                zorder=self.zorder['esdf'],
            )
            plt.colorbar(label="ESDF distance")
            
        if equal: 
            plt.axis("equal")

    def plot_expand_tree(self, expand_tree: Dict[Union[Tuple[int, ...], Tuple[float, ...]], Node], 
                        node_color: str = "#8c564b", 
                        edge_color: str = "#e377c2", 
                        node_size: float = 5, 
                        linewidth: float = 1.0, 
                        node_alpha: float = 1.0,
                        edge_alpha: float = 1.0,
                        connect_to_parent: bool = True,
                        map_frame: bool = True) -> None:
        """
        Visualize an expand tree (e.g. RRT).
        
        Args:
            expand_tree: Dict mapping coordinate tuple -> Node (world frame).
            node_color: Color of the nodes.
            edge_color: Color of the edges (parent -> child).
            node_size: Size of node markers.
            linewidth: Line width of edges.
            connect_to_parent: Whether to draw parent-child connections.
            map_frame: whether path is in map frame or not (world frame)
        """
        if not isinstance(expand_tree, list):   # for multiple trees
            expand_tree = [expand_tree]

        for tree in expand_tree:
            for coord, node in tree.items():
                current = node.current
                if map_frame:
                    current = self.grid_map.map_to_world(current)

                self.ax.scatter(current[0], current[1],
                                c=node_color, s=node_size, zorder=self.zorder['expand_tree_node'], alpha=node_alpha)
                if connect_to_parent and node.parent is not None:
                    parent = node.parent
                    if map_frame:
                        parent = self.grid_map.map_to_world(parent)
                    self.ax.plot([parent[0], current[0]],
                                [parent[1], current[1]],
                                color=edge_color, linewidth=linewidth, zorder=self.zorder['expand_tree_edge'], alpha=edge_alpha)

    def plot_path(self, path: List[Union[Tuple[int, ...], Tuple[float, ...]]], 
                    style: str = "-", color: str = "#13ae00", label: str = None, 
                    linewidth: float = 3, marker: str = None, map_frame: bool = True) -> None:
        '''
        Plot path-like information.
        The meaning of parameters are similar to matplotlib.pyplot.plot (https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html).

        Args:
            path: point list of path
            style: style of path 
            color: color of path
            label: label of path
            linewidth: linewidth of path
            marker: marker of path
            map_frame: whether path is in map frame or not (world frame)
        '''
        if len(path) == 0:
            return

        if map_frame:
            path = [self.grid_map.map_to_world(point) for point in path]

        path = np.array(path)
        
        self.ax.plot(path[:, 0], path[:, 1], style, lw=linewidth, color=color, label=label, marker=marker, zorder=self.zorder['path_2d'])

        if label:
            self.ax.legend()

    def plot_circular_robot(self, robot: CircularRobot, axis_equal: bool = True) -> None:
        """
        Plot a circular robot.

        Args:
            robot: CircularRobot object.
            axis_equal: Whether to set equal aspect ratio for x and y axes.
        """
        patch = plt.Circle(tuple(robot.pos), robot.radius, 
            color=robot.color, alpha=robot.alpha, fill=robot.fill, 
            linewidth=robot.linewidth, linestyle=robot.linestyle,
            zorder=self.zorder['robot_circle'])
        self.ax.add_patch(patch)

        fontsize = robot.fontsize if robot.fontsize else robot.radius * 10
        text = self.ax.text(*robot.pos, robot.text, color=robot.text_color, ha='center', va='center', 
                            fontsize=fontsize, zorder=self.zorder['robot_text'])

        theta = robot.orient[0]
        dx = np.cos(theta) * robot.radius
        dy = np.sin(theta) * robot.radius
        orient_patch = self.ax.arrow(robot.pos[0], robot.pos[1], dx, dy,
                                        head_width=0.1*robot.radius, head_length=0.2*robot.radius,
                                        fc=robot.color, ec=robot.text_color, zorder=self.zorder['robot_orient'])
        return patch, text, orient_patch

    def render_toy_simulator(self, env: ToySimulator, controllers: Dict[str, BaseController],
            steps: int = 1000, interval: int = None,
            show_traj: bool = True, traj_kwargs: dict = {"linestyle": '-', "alpha": 0.7, "linewidth": 1.5},
            show_env_info: bool = False, rtf_limit: float = 1.0, grid_kwargs: dict = {},
            show_pred_traj: bool = True) -> None:
        """
        Render the toy simulator.

        Args:
            env: ToySimulator object.
            controllers: Controllers for each robot.
            steps: Number of steps to render.
            interval: Interval between frames in milliseconds.
            show_traj: Whether to show trajectories.
            traj_kwargs: Keyword arguments for trajectories.
            show_env_info: Whether to show environment information.
            rtf_limit: Maximum real-time factor.
            grid_kwargs: Keyword arguments for grid map.
            show_pred_traj: Whether to show predicted trajectories (for DWA etc.).
        """

        if interval is None:
            interval = int(1000 * env.dt)

        if traj_kwargs.get("color") is None:
            traj_color = {rid: robot.color for rid, robot in env.robots.items()}
        else:
            traj_color = {rid: traj_kwargs.get("color") for rid, robot in env.robots.items()}

        # Draw static map and paths
        self.ax.clear()
        self.plot_grid_map(env.obstacle_grid, **grid_kwargs)

        self.trajs = {rid: {
            "poses": [],
            "time": []
        } for rid in env.robots}

        last_time = time.time()
        if show_env_info:
            env_info_text = self.ax.text(0.02, 0.95, "", transform=self.ax.transAxes, 
                                        ha="left", va="top", alpha=0.5, color="white", zorder=self.zorder['env_info_text'])
            env_info_text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])

        prepare_frames = 5

        def update(frame):
            nonlocal last_time
            nonlocal prepare_frames

            patches = []
            actions = {}

            while prepare_frames > 0:    # matplotlib has a bug
                prepare_frames -= 1
                return patches

            for rid, robot in env.robots.items():
                self.trajs[rid]["poses"].append(robot.pose.copy())
                self.trajs[rid]["time"].append(env.time)

                ob = robot.get_observation(env)
                act, lookahead_pose = controllers[rid].get_action(ob)

                if lookahead_pose is not None:
                    lookahead_pose_patch = plt.Circle(lookahead_pose[:2], 0.2, color=robot.color, alpha=0.5, zorder=self.zorder['lookahead_pose_node'])
                    self.ax.add_patch(lookahead_pose_patch)
                    patches.append(lookahead_pose_patch)

                    theta = lookahead_pose[2]
                    dx = np.cos(theta) * robot.radius
                    dy = np.sin(theta) * robot.radius
                    orient_patch = self.ax.arrow(lookahead_pose[0], lookahead_pose[1], dx, dy,
                                                width=0.2*robot.radius,
                                                fc=robot.color, ec=robot.color, alpha=0.5, zorder=self.zorder['lookahead_pose_orient'])
                    patches.append(orient_patch)

                actions[rid] = act

            for rid, robot in env.robots.items():
                items = self.plot_circular_robot(robot)
                for item in items:
                    if item is not None:
                        patches.append(item)

            # draw trajectories
            if show_traj:
                for rid, traj in self.trajs.items():
                    poses = traj["poses"]
                    if len(poses) > 1:
                        pose_x = [p[0] for p in poses]
                        pose_y = [p[1] for p in poses]
                        traj_line, = self.ax.plot(pose_x, pose_y, color=traj_color[rid], zorder=self.zorder['traj'], **traj_kwargs)
                        patches.append(traj_line)

            if show_pred_traj:
                for rid, controller in controllers.items():
                    pred_traj = controller.pred_traj
                    if len(pred_traj) > 1:
                        pred_traj_x = [p[0] for p in pred_traj]
                        pred_traj_y = [p[1] for p in pred_traj]
                        pred_traj_line, = self.ax.plot(pred_traj_x, pred_traj_y, color=traj_color[rid], zorder=self.zorder['pred_traj'], **traj_kwargs)
                        patches.append(pred_traj_line)

            elapsed = time.time() - last_time
            if rtf_limit and env.dt / elapsed > rtf_limit:
                time.sleep(env.dt / rtf_limit - elapsed)
                elapsed = time.time() - last_time

            if show_env_info:
                step_count = env.step_count
                sim_time = step_count * env.dt
                rtf = env.dt / elapsed
                env_info_text.set_text(f"Step: {step_count}, Time: {sim_time:.3f}s, RTF: {rtf:.3f}")
                patches.append(env_info_text)

            last_time = time.time()

            if env.step_count < steps:
                obs, rewards, dones, info = env.step(actions)

            return patches

        self.ani = animation.FuncAnimation(
            self.fig, update, frames=steps+prepare_frames, interval=interval, blit=True, repeat=False
        )

    def set_title(self, title: str) -> None:
        """
        Set title.

        Args:
            title: Title.
        """
        plt.title(title)

    def connect(self, name: str, func) -> None:
        """
        Connect event.

        Args:
            name: Event name.
            func: Event function.
        """
        self.fig.canvas.mpl_connect(name, func)

    def clean(self):
        """
        Clean plot.
        """
        plt.cla()

    def update(self):
        """
        Update plot.
        """
        self.fig.canvas.draw_idle()

    def savefig(self, filename, *args, **kwargs):
        """
        Save figure. 

        Args:
            filename: Filename to save.
            *args: For 2D, see matplotlib.pyplot.savefig. For 3D, see pyvista.Plotter.screenshot.
            **kwargs: For 2D, see matplotlib.pyplot.savefig. For 3D, see pyvista.Plotter.screenshot.
        """
        plt.savefig(fname=filename, *args, **kwargs)

    def show(self):
        """
        Show plot.
        """
        plt.show()

    def legend(self):
        """
        Add legend.
        """
        plt.legend()
    
    def close(self):
        """
        Close plot.
        """
        plt.close()
