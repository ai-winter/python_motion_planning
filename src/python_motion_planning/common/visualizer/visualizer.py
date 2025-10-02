"""
@file: visualization.py
@breif: visualization
@author: Yang Haodong, Wu Maojia
@update: 2025.9.20
"""
from typing import Union, Dict, List, Tuple, Any
from collections import namedtuple
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import animation
import matplotlib.patheffects as path_effects

from python_motion_planning.controller import BaseController
from python_motion_planning.common.env import TYPES, ToySimulator, Grid, CircularRobot, Node
from python_motion_planning.common.utils import Geometry

class Visualizer:
    def __init__(self, figname: str = "", figsize: tuple = (10, 8)):
        self.fig = plt.figure(figname, figsize=figsize)
        self.ax = self.fig.add_subplot()
        self.ani = None

        # colors
        self.cmap_dict = {
            TYPES.FREE: "#ffffff",
            TYPES.OBSTACLE: "#000000",
            TYPES.START: "#ff0000",
            TYPES.GOAL: "#1155cc",
            TYPES.INFLATION: "#ffccff",
            TYPES.EXPAND: "#eeeeee",
            TYPES.CUSTOM: "#bbbbbb",
        }
        # self.norm = mcolors.BoundaryNorm(list(range(len(self.cmap_dict))), len(self.cmap_dict))

        self.zorder = {
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

        self.cmap = mcolors.ListedColormap([info for info in self.cmap_dict.values()])
        self.norm = mcolors.BoundaryNorm([i for i in range(self.cmap.N + 1)], self.cmap.N)
        self.grid_map = None
        self.dim = None

        self.trajs = {}

    def __del__(self):
        self.close()

    def plot_grid_map(self, grid_map: Grid, equal: bool = False, alpha_3d: dict = {
                            TYPES.FREE: 0.0,
                            TYPES.OBSTACLE: 0.5,
                            TYPES.START: 0.5,
                            TYPES.GOAL: 0.5,
                            TYPES.INFLATION: 0.0,
                            TYPES.EXPAND: 0.1,
                            TYPES.CUSTOM: 0.5,
                        },
                        show_esdf: bool = False, alpha_esdf: float = 0.5) -> None:
        '''
        Plot grid map with static obstacles.

        Args:
            map: Grid map or its type map.
            equal: Whether to set axis equal.
            alpha_3d: Alpha of occupancy for 3d visualization.
            show_esdf: Whether to show esdf.
            alpha_esdf: Alpha of esdf.
        '''
        self.grid_map = grid_map
        self.dim = grid_map.dim
        if grid_map.dim == 2:
            plt.imshow(
                np.transpose(grid_map.type_map.array), 
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

        elif grid_map.dim == 3:
            self.ax = self.fig.add_subplot(projection='3d')

            data = grid_map.type_map.array
            nx, ny, nz = data.shape

            filled = np.zeros_like(data, dtype=bool)
            colors = np.zeros(data.shape + (4,), dtype=float)  # RGBA

            for key, color in self.cmap_dict.items():
                mask = (data == key)
                if alpha_3d[key] < 1e-6:
                    continue
                filled |= mask
                rgba = matplotlib.colors.to_rgba(color, alpha=alpha_3d[key])
                colors[mask] = rgba

            self.ax.voxels(filled, facecolors=colors, zorder=self.zorder['voxels'])

            if show_esdf:
                # TODO
                raise NotImplementedError

            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.set_zlabel("Z")

            # let voxels look not stretched
            max_range = 0
            for d in range(grid_map.dim):
                max_range = max(max_range, grid_map.bounds[d, 1] - grid_map.bounds[d, 0])
            self.ax.set_xlim(grid_map.bounds[0, 0], grid_map.bounds[0, 0] + max_range)
            self.ax.set_ylim(grid_map.bounds[1, 0], grid_map.bounds[1, 0] + max_range)
            self.ax.set_zlim(grid_map.bounds[2, 0], grid_map.bounds[2, 0] + max_range)

            if equal:
                self.ax.set_box_aspect([1,1,1])

        else:
            raise NotImplementedError(f"Grid map with dim={grid_map.dim} not supported.")

    def plot_expand_tree(self, expand_tree: Dict[Union[Tuple[int, ...], Tuple[float, ...]], Node], 
                        node_color: str = "C5", 
                        edge_color: str = "C6", 
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
        if self.dim == 2:
            for coord, node in expand_tree.items():
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

        elif self.dim == 3:
            for coord, node in expand_tree.items():
                current = node.current
                if map_frame:
                    current = self.grid_map.map_to_world(current)

                self.ax.scatter(current[0], current[1], current[2],
                                c=node_color, s=node_size, zorder=self.zorder['expand_tree_node'], alpha=node_alpha)
                if connect_to_parent and node.parent is not None:
                    parent = node.parent
                    if map_frame:
                        parent = self.grid_map.map_to_world(parent)
                    self.ax.plot([parent[0], current[0]],
                                [parent[1], current[1]],
                                [parent[2], current[2]],
                                color=edge_color, linewidth=linewidth, zorder=self.zorder['expand_tree_edge'], alpha=edge_alpha)

        else:
            raise ValueError("Dimension must be 2 or 3")


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
        
        if self.dim == 2:
            self.ax.plot(path[:, 0], path[:, 1], style, lw=linewidth, color=color, label=label, marker=marker, zorder=self.zorder['path_2d'])
        elif self.dim == 3:
            self.ax.plot(path[:, 0], path[:, 1], path[:, 2], style, lw=linewidth, color=color, label=label, marker=marker, zorder=self.zorder['path_3d'])
        else:
            raise ValueError("Dimension not supported")

        if label:
            self.ax.legend()

    def plot_circular_robot(self, robot: CircularRobot, axis_equal: bool = True) -> None:
        patch = plt.Circle(tuple(robot.pos), robot.radius, 
            color=robot.color, alpha=robot.alpha, fill=robot.fill, 
            linewidth=robot.linewidth, linestyle=robot.linestyle,
            zorder=self.zorder['robot_circle'])
        self.ax.add_patch(patch)

        fontsize = robot.fontsize if robot.fontsize else robot.radius * 10
        text = self.ax.text(*robot.pos, robot.text, color=robot.text_color, ha='center', va='center', 
                            fontsize=fontsize, zorder=self.zorder['robot_text'])

        if robot.dim == 2:
            theta = robot.orient[0]
            dx = np.cos(theta) * robot.radius
            dy = np.sin(theta) * robot.radius
            orient_patch = self.ax.arrow(robot.pos[0], robot.pos[1], dx, dy,
                                         head_width=0.1*robot.radius, head_length=0.2*robot.radius,
                                         fc=robot.color, ec=robot.text_color, zorder=self.zorder['robot_orient'])
            return patch, text, orient_patch
        elif robot.dim == 3:
            # TODO: quiver for 3D vector
            return patch, text
        else:
            return patch, text

    def render_toy_simulator(self, env: ToySimulator, controllers: Dict[str, BaseController],
            steps: int = 1000, interval: int = None,
            show_traj: bool = True, traj_kwargs: dict = {"linestyle": '-', "alpha": 0.7, "linewidth": 1.5},
            show_env_info: bool = False, rtf_limit: float = 1.0, grid_kwargs: dict = {},
            show_pred_traj: bool = True) -> None:

        if interval is None:
            interval = int(1000 * env.dt)

        if traj_kwargs.get("color") is None:
            traj_color = {rid: robot.color for rid, robot in env.robots.items()}
        else:
            traj_color = {rid: traj_kwargs.get("color") for rid, robot in env.robots.items()}

        # 先画静态的地图和路径
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

    def get_traj_info(self, rid: int, goal_pose: np.ndarray, goal_dist_tol: float, goal_orient_tol: float) -> Dict[str, Any]:
        traj = self.trajs[rid]

        info = {
            "traj_length": 0.0,
            "success": False,
            "dist_success": False, 
            "oracle_success": False,
            "oracle_dist_success": False,
            "success_time": None,
            "dist_success_time": None,
            "oracle_success_time": None,
            "oracle_dist_success_time": None,
        }

        for i in range(len(traj["poses"])):
            pose = traj["poses"][i]
            time = traj["time"][i]
            
            pos = pose[:self.dim]
            orient = pose[self.dim:]
            goal_pos = goal_pose[:self.dim]
            goal_orient = goal_pose[self.dim:]

            if i > 0:
                info["traj_length"] += np.linalg.norm(pos - traj["poses"][i-1][:self.dim])

            if np.linalg.norm(pos - goal_pos) < goal_dist_tol:
                if not info["oracle_dist_success"]:
                    info["oracle_dist_success"] = True
                    info["oracle_dist_success_time"] = time

                if not info["dist_success"]:
                    info["dist_success"] = True
                    info["dist_success_time"] = time

                if np.abs(Geometry.regularize_orient(orient - goal_orient)) < goal_orient_tol:
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

        info["traj_length"] = float(info["traj_length"])
        return info


    def set_title(self, title: str) -> None:
        plt.title(title)

    def connect(self, name: str, func) -> None:
        self.fig.canvas.mpl_connect(name, func)

    def clean(self):
        plt.cla()

    def update(self):
        self.fig.canvas.draw_idle()
    
    def show(self):
        plt.show()

    def legend(self):
        plt.legend()
    
    def close(self):
        plt.close()
