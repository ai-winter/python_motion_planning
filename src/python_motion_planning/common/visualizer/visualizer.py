"""
@file: visualization.py
@breif: visualization
@author: Yang Haodong, Wu Maojia
@update: 2025.9.20
"""
from typing import Union, Dict
from collections import namedtuple
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import animation

from python_motion_planning.controller import BaseController
from python_motion_planning.common.env import TYPES, ToySimulator, Grid, CircularRobot, BallRobot

class Visualizer:
    def __init__(self, fig_name: str = ""):
        self.fig = plt.figure(fig_name)
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

        self.cmap = mcolors.ListedColormap([info for info in self.cmap_dict.values()])
        self.norm = mcolors.BoundaryNorm([i for i in range(self.cmap.N + 1)], self.cmap.N)
        self.grid_map = None

    def plot_grid_map(self, grid_map: Grid, equal: bool = False, alpha_3d: float = 0.1,
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
        if grid_map.dim == 2:
            plt.imshow(
                np.transpose(grid_map.type_map.array), 
                cmap=self.cmap, 
                norm=self.norm, 
                origin='lower', 
                interpolation='nearest', 
                extent=[*grid_map.bounds[0], *grid_map.bounds[1]]
                )

            if show_esdf:   # draw esdf hotmap
                plt.imshow(
                    np.transpose(grid_map.esdf),
                    cmap="jet",
                    origin="lower",
                    interpolation="nearest",
                    extent=[*grid_map.bounds[0], *grid_map.bounds[1]],
                    alpha=alpha_esdf
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
                if key == TYPES.FREE:
                    continue
                filled |= mask
                rgba = matplotlib.colors.to_rgba(color, alpha=alpha_3d)  # (r,g,b,a)
                colors[mask] = rgba

            self.ax.voxels(filled, facecolors=colors)

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

    def plot_path(self, path: list, style: str = "-", color: str = "#13ae00", label: str = None, linewidth: float = 2, marker: str = None) -> None:
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
        '''
        path = np.array(path)
        if len(path.shape) < 2:
            return
        if path.shape[1] == 2:
            plt.plot(path[:, 0], path[:, 1], style, lw=linewidth, color=color, label=label, marker=marker)
        elif path.shape[1] == 3:
            self.ax.plot(path[:, 0], path[:, 1], path[:, 2], style, lw=linewidth, color=color, label=label, marker=marker)
        else:
            raise ValueError("Path dimension not supported")

        if label:
            self.ax.legend()

    def plot_circular_robot(self, robot: CircularRobot, axis_equal: bool = True) -> None:
        patch = plt.Circle(tuple(robot.pos), robot.radius, 
            color=robot.color, alpha=robot.alpha, fill=robot.fill, linewidth=robot.linewidth, linestyle=robot.linestyle)
        self.ax.add_patch(patch)

        fontsize = robot.fontsize if robot.fontsize else robot.radius * 15
        text = self.ax.text(*robot.pos, robot.text, color=robot.text_color, ha='center', va='center', fontsize=fontsize)

        # === 新增：绘制朝向 ===
        if robot.dim == 2:
            theta = robot.orient[0]
            dx = np.cos(theta) * robot.radius
            dy = np.sin(theta) * robot.radius
            orient_patch = self.ax.arrow(robot.pos[0], robot.pos[1], dx, dy,
                                         head_width=0.1*robot.radius, head_length=0.2*robot.radius,
                                         fc=robot.color, ec=robot.text_color)
            return patch, text, orient_patch
        elif robot.dim == 3:
            # TODO: 可以用 quiver 绘制 3D 方向向量
            return patch, text
        else:
            return patch, text

    def render_toy_simulator(self, env: ToySimulator, controllers: Dict[str, BaseController], steps: int = 1000, interval: int = 50,
            show_traj: bool = True, traj_kwargs: dict = {"linestyle": '-', "alpha": 0.7, "linewidth": 1.5},
            show_env_info: bool = False, limit_rtf: bool = True, grid_kwargs: dict = {},
            show_pred_traj: bool = True) -> None:

        if traj_kwargs.get("color") is None:
            traj_color = {rid: robot.color for rid, robot in env.robots.items()}
        else:
            traj_color = {rid: traj_kwargs.get("color") for rid, robot in env.robots.items()}

        # 先画静态的地图和路径
        self.ax.clear()
        self.plot_grid_map(env.obstacle_grid, **grid_kwargs)

        trajectories = {rid: [] for rid in env.robots}

        last_time = time.time()
        if show_env_info:
            env_info_text_black = self.ax.text(0.02, 0.95, "", transform=self.ax.transAxes, ha="left", va="top", alpha=0.5, color="black")
            env_info_text_white = self.ax.text(0.02, 0.95, "", transform=self.ax.transAxes, ha="left", va="top", alpha=0.5, color="white")

        def update(frame):
            nonlocal last_time

            # 每帧只更新机器人，不清理整个画布
            patches = []
            actions = {}
            for rid, robot in env.robots.items():
                trajectories[rid].append(robot.pos.copy())

                ob = robot.get_observation(env)
                act, lookahead_pose = controllers[rid].get_action(ob)

                if lookahead_pose is not None:
                    lookahead_pose_patch = plt.Circle(lookahead_pose[:2], 0.2, color=robot.color, alpha=0.5)
                    self.ax.add_patch(lookahead_pose_patch)
                    patches.append(lookahead_pose_patch)

                    theta = lookahead_pose[2]
                    dx = np.cos(theta) * robot.radius
                    dy = np.sin(theta) * robot.radius
                    orient_patch = self.ax.arrow(lookahead_pose[0], lookahead_pose[1], dx, dy,
                                                width=0.2*robot.radius,
                                                fc=robot.color, ec=robot.color, alpha=0.5)
                    patches.append(orient_patch)

                actions[rid] = act

            obs, rewards, dones, info = env.step(actions)

            for rid, robot in env.robots.items():
                items = self.plot_circular_robot(robot)
                for item in items:
                    if item is not None:
                        patches.append(item)

            # draw trajectories
            if show_traj:
                for rid, traj in trajectories.items():
                    if len(traj) > 1:
                        traj_x = [p[0] for p in traj]
                        traj_y = [p[1] for p in traj]
                        traj_line, = self.ax.plot(traj_x, traj_y, color=traj_color[rid], **traj_kwargs)
                        patches.append(traj_line)

            if show_pred_traj:
                for rid, controller in controllers.items():
                    pred_traj = controller.pred_traj
                    if len(pred_traj) > 1:
                        pred_traj_x = [p[0] for p in pred_traj]
                        pred_traj_y = [p[1] for p in pred_traj]
                        pred_traj_line, = self.ax.plot(pred_traj_x, pred_traj_y, color=traj_color[rid], **traj_kwargs)
                        patches.append(pred_traj_line)

            elapsed = time.time() - last_time
            if limit_rtf and elapsed < env.dt:
                time.sleep(env.dt - elapsed)
                elapsed = time.time() - last_time

            if show_env_info:
                sim_time = env.step_count * env.dt
                rtf = env.dt / elapsed
                env_info_text_black.set_text(f"Step: {env.step_count}, Time: {sim_time:.3f}s, RTF: {rtf:.3f}")
                env_info_text_white.set_text(f"Step: {env.step_count}, Time: {sim_time:.3f}s, RTF: {rtf:.3f}")
                patches.append(env_info_text_black)
                patches.append(env_info_text_white)

            last_time = time.time()

            return patches

        self.ani = animation.FuncAnimation(
            self.fig, update, frames=steps, interval=interval, blit=True, repeat=False
        )

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



# from typing import Union, Dict
# from collections import namedtuple

# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# from matplotlib import animation

# from python_motion_planning.controller import BaseController
# from python_motion_planning.common.env import TYPES, ToySimulator, Grid, CircularRobot, BallRobot

# class Visualizer:
#     def __init__(self, fig_name: str = ""):
#         self.fig = plt.figure(fig_name)
#         self.ax = self.fig.add_subplot()
#         self.ani = None

#         # colors
#         self.cmap_dict = {
#             TYPES.FREE: "#ffffff",
#             TYPES.OBSTACLE: "#000000",
#             TYPES.START: "#ff0000",
#             TYPES.GOAL: "#1155cc",
#             TYPES.INFLATION: "#ffccff",
#             TYPES.EXPAND: "#eeeeee",
#             TYPES.CUSTOM: "#bbbbbb",
#         }
#         # self.norm = mcolors.BoundaryNorm(list(range(len(self.cmap_dict))), len(self.cmap_dict))

#         self.cmap = mcolors.ListedColormap([info for info in self.cmap_dict.values()])
#         self.norm = mcolors.BoundaryNorm([i for i in range(self.cmap.N + 1)], self.cmap.N)
#         self.grid_map = None

#     def plot_grid_map(self, grid_map: Grid, equal: bool = False, alpha: float = 0.1) -> None:
#         '''
#         Plot grid map with static obstacles.

#         Args:
#             map: Grid map or its type map.
#             equal: Whether to set axis equal.
#             alpha: Alpha of occupancy for 3d visualization.
#         '''
#         if grid_map.dim == 2:
#             plt.imshow(np.transpose(grid_map.type_map.array), cmap=self.cmap, norm=self.norm, origin='lower', interpolation='nearest', 
#                 extent=[*grid_map.bounds[0], *grid_map.bounds[1]])
#             if equal: 
#                 plt.axis("equal")

#         elif grid_map.dim == 3:
#             self.ax = self.fig.add_subplot(projection='3d')

#             data = grid_map.type_map.array
#             nx, ny, nz = data.shape

#             filled = np.zeros_like(data, dtype=bool)
#             colors = np.zeros(data.shape + (4,), dtype=float)  # RGBA

#             for key, color in self.cmap_dict.items():
#                 mask = (data == key)
#                 if key == TYPES.FREE:
#                     continue
#                 filled |= mask
#                 rgba = matplotlib.colors.to_rgba(color, alpha=alpha)  # (r,g,b,a)
#                 colors[mask] = rgba

#             self.ax.voxels(filled, facecolors=colors)

#             self.ax.set_xlabel("X")
#             self.ax.set_ylabel("Y")
#             self.ax.set_zlabel("Z")

#             # let voxels look not stretched
#             max_range = 0
#             for d in range(grid_map.dim):
#                 max_range = max(max_range, grid_map.bounds[d, 1] - grid_map.bounds[d, 0])
#             self.ax.set_xlim(grid_map.bounds[0, 0], grid_map.bounds[0, 0] + max_range)
#             self.ax.set_ylim(grid_map.bounds[1, 0], grid_map.bounds[1, 0] + max_range)
#             self.ax.set_zlim(grid_map.bounds[2, 0], grid_map.bounds[2, 0] + max_range)

#             if equal:
#                 self.ax.set_box_aspect([1,1,1])

#         else:
#             raise NotImplementedError(f"Grid map with dim={grid_map.dim} not supported.")

#     def set_title(self, title: str) -> None:
#         plt.title(title)

#     def plot_path(self, path: list, style: str = "-", color: str = "#13ae00", label: str = None, linewidth: float = 2, marker: str = None) -> None:
#         '''
#         Plot path-like information.
#         The meaning of parameters are similar to matplotlib.pyplot.plot (https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html).

#         Args:
#             path: point list of path
#             style: style of path
#             color: color of path
#             label: label of path
#             linewidth: linewidth of path
#             marker: marker of path
#         '''
#         path = np.array(path)
#         if path.shape[1] == 2:
#             plt.plot(path[:, 0], path[:, 1], style, lw=linewidth, color=color, label=label, marker=marker)
#         elif path.shape[1] == 3:
#             self.ax.plot(path[:, 0], path[:, 1], path[:, 2], style, lw=linewidth, color=color, label=label, marker=marker)
#         else:
#             raise ValueError("Path dimension not supported")

#         if label:
#             self.ax.legend()

#     def plot_circular_robot(self, robot: CircularRobot, axis_equal: bool = True) -> None:
#         patch = plt.Circle(tuple(robot.pos), robot.radius, 
#             color=robot.color, alpha=robot.alpha, fill=robot.fill, linewidth=robot.linewidth, linestyle=robot.linestyle)
#         self.ax.add_patch(patch)

#         fontsize = robot.fontsize if robot.fontsize else robot.radius * 15

#         text = self.ax.text(*robot.pos, robot.text, color=robot.text_color, ha='center', va='center', fontsize=fontsize)
#         return patch, text

#     def render_toy_simulator(self, env: ToySimulator, controllers: Dict[str, BaseController], steps: int = 1000, interval: int = 50,
#             show_traj: bool = True, traj_style: str = '-', traj_color: Dict[str, str] = None, traj_alpha: float = 0.7, traj_width = 1.5) -> None:
#         if traj_color is None:
#             traj_color = {rid: robot.color for rid, robot in env.robots.items()}

#         # 先画静态的地图和路径
#         self.ax.clear()
#         self.plot_grid_map(env.obstacle_grid)

#         trajectories = {rid: [] for rid in env.robots}

#         def update(frame):
#             # 每帧只更新机器人，不清理整个画布
#             patches = []
#             texts = []
#             actions = {}
#             for rid, robot in env.robots.items():
#                 trajectories[rid].append(robot.pos.copy())

#                 ob = robot.get_observation(env)
#                 act, lookahead_pt = controllers[rid].get_action(ob)

#                 if lookahead_pt is not None:
#                     lookahead_pt_patch = plt.Circle(lookahead_pt, 0.2, color=robot.color, alpha=0.5)
#                     self.ax.add_patch(lookahead_pt_patch)
#                     patches.append(lookahead_pt_patch)

#                 actions[rid] = act

#             obs, rewards, dones, info = env.step(actions)

#             for rid, robot in env.robots.items():
#                 p, t = self.plot_circular_robot(robot)
#                 patches.append(p)
#                 texts.append(t)

#             # draw trajectories
#             if show_traj:
#                 for rid, traj in trajectories.items():
#                     if len(traj) > 1:
#                         traj_x = [p[0] for p in traj]
#                         traj_y = [p[1] for p in traj]
#                         traj_line, = self.ax.plot(traj_x, traj_y, traj_style, color=traj_color[rid], alpha=traj_alpha, linewidth=traj_width)
#                         patches.append(traj_line)

#             return patches + texts

#         self.ani = animation.FuncAnimation(
#             self.fig, update, frames=steps, interval=interval, blit=True, repeat=False
#         )

#     def connect(self, name: str, func) -> None:
#         self.fig.canvas.mpl_connect(name, func)

#     def clean(self):
#         plt.cla()

#     def update(self):
#         self.fig.canvas.draw_idle()
    
#     def show(self):
#         plt.show()

#     def legend(self):
#         plt.legend()



    # def plotMarkers(self, markers: list, axis_equal: bool=True, name: str="normal", props: dict={}) -> None:
    #     '''
    #     Plot markers.
    #     '''
    #     color = props["color"] if "color" in props.keys() else "#13ae00"
    #     label = props["label"] if "label" in props.keys() else None
    #     size = props["size"] if "size" in props.keys() else 2
    #     marker = props["marker"] if "marker" in props.keys() else None

    #     if name == "normal":
    #         plt.scatter(markers[0], markers[1], s=size, c=color, marker=marker)
    #     elif name == "arrow":
    #         length = props["length"] if "length" in props.keys() else 2
    #         for x, y, theta in markers:
    #             self.plotArrow(x, y, theta, length, color)
    #     else:
    #         raise NotImplementedError
        
    #     if axis_equal:
    #         plt.axis("equal")
        
    #     if label:
    #         plt.legend()

    # def plotFrames(self, frame_info: list) -> None:
    #     marker_dict, line_dict = {}, {}
    #     frame_num = max([len(info["data"]) for info in frame_info])
    #     for i in range(frame_num):
    #         for j, info in enumerate(frame_info):
    #             idx = i if i < len(info["data"]) else -1
    #             props = info["props"] if "props" in info.keys() else None
    #             # agent
    #             if info["name"] == "agent":
    #                 color = props["color"] if props is not None and "color" in props.keys() else "r"
    #                 radius = props["radius"] if props is not None and "radius" in props.keys() else 1.0
    #                 self.plotAgent(info["data"][idx], radius, color)
    #             # marker
    #             if info["name"] == "marker":
    #                 resume = props["resume"] if props is not None and "resume" in props.keys() else True
    #                 color = props["color"] if props is not None and "color" in props.keys() else "b"
    #                 style = props["style"] if props is not None and "style" in props.keys() else "o"
    #                 size = props["size"] if props is not None and "size" in props.keys() else 50
    #                 if resume and f"marker_{j}" in marker_dict.keys():
    #                     marker_dict[f"marker_{j}"].remove()
    #                 marker_dict[f"marker_{j}"] = self.ax.scatter(
    #                     info["data"][idx][0], info["data"][idx][1], c=color, s=size, marker=style
    #                 )
    #             # line
    #             if info["name"] == "line":
    #                 resume = props["resume"] if props is not None and "resume" in props.keys() else True
    #                 color = props["color"] if props is not None and "color" in props.keys() else "#13ae00"
    #                 style = props["style"] if props is not None and "style" in props.keys() else "-"
    #                 width = props["width"] if props is not None and "width" in props.keys() else 2
    #                 for k, line in enumerate(info["data"][idx]):
    #                     if resume and f"line_{j}_{k}" in line_dict.keys():
    #                         line_dict[f"line_{j}_{k}"].pop(0).remove()
    #                     line_dict[f"line_{j}_{k}"] = self.ax.plot(line[0], line[1], style, c=color, lw=width)    
                
    #         plt.gcf().canvas.mpl_connect('key_release_event',
    #                     lambda event: [exit(0) if event.key == 'escape' else None])
    #         if i % 5 == 0:             plt.pause(0.03)


    # def plotAgent(self, pose: tuple, radius: float=1, color: str="#f00") -> None:
    #     '''
    #     Plot agent with specifical pose.

    #     Args
    #     ----------
    #     pose: Pose of agent
    #     radius: Radius of agent
    #     '''
    #     x, y, theta = pose
    #     ref_vec = np.array([[radius / 2], [0]])
    #     rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
    #                         [np.sin(theta),  np.cos(theta)]])
    #     end_pt = rot_mat @ ref_vec + np.array([[x], [y]])

    #     try:
    #         self.ax.artists.pop()
    #         for art in self.ax.get_children():
    #             if isinstance(art, matplotlib.patches.FancyArrow):
    #                 art.remove()
    #     except:
    #         pass

    #     self.ax.arrow(x, y, float(end_pt[0]) - x, float(end_pt[1]) - y,
    #             width=0.1, head_width=0.40, color=color)
    #     circle = plt.Circle((x, y), radius, color=color, fill=False)
    #     self.ax.add_artist(circle)

    # def plotCurve(self, xlist: list, ylist: list, xlabels: list, ylabels: list,
    #     color: str=None, rows: int=1, cols: int=1, fig_name: str=None) -> None:
    #     if fig_name:
    #         plt.figure(fig_name)
        
    #     nums = rows * cols
    #     for i in range(nums):
    #         plt.subplot(rows, cols, i + 1)
    #         plt.plot(xlist[i], ylist[i], c=color)
    #         plt.xlabel(xlabels[i])
    #         plt.ylabel(ylabels[i])
    #         plt.title(fig_name)

    # def plotArrow(self, x, y, theta, length, color):
    #     angle = np.deg2rad(30)
    #     d = 0.5 * length
    #     w = 2

    #     x_start, y_start = x, y
    #     x_end = x + length * np.cos(theta)
    #     y_end = y + length * np.sin(theta)

    #     theta_hat_L = theta + np.pi - angle
    #     theta_hat_R = theta + np.pi + angle

    #     x_hat_start = x_end
    #     x_hat_end_L = x_hat_start + d * np.cos(theta_hat_L)
    #     x_hat_end_R = x_hat_start + d * np.cos(theta_hat_R)

    #     y_hat_start = y_end
    #     y_hat_end_L = y_hat_start + d * np.sin(theta_hat_L)
    #     y_hat_end_R = y_hat_start + d * np.sin(theta_hat_R)

    #     plt.plot([x_start, x_end], [y_start, y_end], color=color, linewidth=w)
    #     plt.plot([x_hat_start, x_hat_end_L], [y_hat_start, y_hat_end_L], color=color, linewidth=w)
    #     plt.plot([x_hat_start, x_hat_end_R], [y_hat_start, y_hat_end_R], color=color, linewidth=w)

    # @staticmethod
    # def plotCar(x, y, theta, width, length, color):
    #     theta_B = np.pi + theta

    #     xB = x + length / 4 * np.cos(theta_B)
    #     yB = y + length / 4 * np.sin(theta_B)

    #     theta_BL = theta_B + np.pi / 2
    #     theta_BR = theta_B - np.pi / 2

    #     x_BL = xB + width / 2 * np.cos(theta_BL)        # Bottom-Left vertex
    #     y_BL = yB + width / 2 * np.sin(theta_BL)
    #     x_BR = xB + width / 2 * np.cos(theta_BR)        # Bottom-Right vertex
    #     y_BR = yB + width / 2 * np.sin(theta_BR)

    #     x_FL = x_BL + length * np.cos(theta)               # Front-Left vertex
    #     y_FL = y_BL + length * np.sin(theta)
    #     x_FR = x_BR + length * np.cos(theta)               # Front-Right vertex
    #     y_FR = y_BR + length * np.sin(theta)

    #     plt.plot([x_BL, x_BR, x_FR, x_FL, x_BL],
    #              [y_BL, y_BR, y_FR, y_FL, y_BL],
    #              linewidth=1, color=color)

    #     # Plot.plotArrow(x, y, theta, length / 2, color)


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)