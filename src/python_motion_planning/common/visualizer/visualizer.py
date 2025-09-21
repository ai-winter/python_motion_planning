"""
@file: visualization.py
@breif: visualization
@author: Yang Haodong, Wu Maojia
@update: 2025.9.20
"""
from typing import Union, Dict
from collections import namedtuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import animation

from python_motion_planning.controller import BaseController
from python_motion_planning.common.env import TYPES, ToySimulator, Grid, CircularRobot, BallRobot

ColorInfo = namedtuple('ColorInfo', 'idx color')

'''
extra_info = [
    {"type": , "data": , "name": }`
]

type:
    - value
        - cost
        - name
        - success
    - path
        - normal
        - line
    - marker
        - normal
        - arrow
    - grids
        - expand
    - agent
    - callback
    - frames
        - agent
        - line
        - marker
'''

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

    def plotGridMap(self, grid_map: Grid, equal: bool = False) -> None:
        '''
        Plot grid map with static obstacles.

        Parameters:
            map: Grid map or its type map.
            equal: Whether to set axis equal.
        '''
        if grid_map.ndim == 2:
            plt.imshow(np.transpose(grid_map.type_map.array), cmap=self.cmap, norm=self.norm, origin='lower', interpolation='nearest', 
                extent=[*grid_map.bounds[0], *grid_map.bounds[1]])
            if equal: 
                plt.axis("equal")

        elif grid_map.ndim == 3:
            raise NotImplementedError
        else:
            raise NotImplementedError

    def setTitle(self, title: str) -> None:
        plt.title(title)

    def plotPath(self, path: list, style: str = "-", color: str = "#13ae00", label: str = None, linewidth: float = 2, marker: str = None) -> None:
        '''
        Plot path-like information.
        The meaning of parameters are similar to matplotlib.pyplot.plot (https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html).

        Parameters:
            path: point list of path
            style: style of path
            color: color of path
            label: label of path
            linewidth: linewidth of path
            marker: marker of path
        '''
        path_x = [path[i][0] for i in range(len(path))]
        path_y = [path[i][1] for i in range(len(path))]
        plt.plot(path_x, path_y, style, lw=linewidth, color=color, label=label, marker=marker)
        
        if label:
            plt.legend()

    def plotCircularRobot(self, robot: CircularRobot, axis_equal: bool = True) -> None:
        patch = plt.Circle(tuple(robot.pos), robot.radius, 
            color=robot.color, alpha=robot.alpha, fill=robot.fill, linewidth=robot.linewidth, linestyle=robot.linestyle)
        self.ax.add_patch(patch)

        fontsize = robot.fontsize if robot.fontsize else robot.radius * 15

        text = self.ax.text(*robot.pos, robot.text, color=robot.text_color, ha='center', va='center', fontsize=fontsize)
        return patch, text

    def renderToySimulator(self, env: ToySimulator, controllers: Dict[str, BaseController], steps: int = 1000, interval: int = 50,
            show_traj: bool = True, traj_style: str = '-', traj_color: Dict[str, str] = None, traj_alpha: float = 0.7, traj_width = 1.5) -> None:
        if traj_color is None:
            traj_color = {rid: robot.color for rid, robot in env.robots.items()}

        # 先画静态的地图和路径
        self.ax.clear()
        self.plotGridMap(env.obstacle_grid)

        trajectories = {rid: [] for rid in env.robots}

        def update(frame):
            # 每帧只更新机器人，不清理整个画布
            patches = []
            texts = []
            actions = {}
            for rid, robot in env.robots.items():
                trajectories[rid].append(robot.pos.copy())

                ob = robot.get_observation(env)
                act, lookahead_pt = controllers[rid].get_action(ob)

                if lookahead_pt is not None:
                    lookahead_pt_patch = plt.Circle(lookahead_pt, 0.2, color=robot.color, alpha=0.5)
                    self.ax.add_patch(lookahead_pt_patch)
                    patches.append(lookahead_pt_patch)

                actions[rid] = act

            obs, rewards, dones, info = env.step(actions)

            for rid, robot in env.robots.items():
                p, t = self.plotCircularRobot(robot)
                patches.append(p)
                texts.append(t)

            # draw trajectories
            if show_traj:
                for rid, traj in trajectories.items():
                    if len(traj) > 1:
                        traj_x = [p[0] for p in traj]
                        traj_y = [p[1] for p in traj]
                        traj_line, = self.ax.plot(traj_x, traj_y, traj_style, color=traj_color[rid], alpha=traj_alpha, linewidth=traj_width)
                        patches.append(traj_line)

            return patches + texts

        self.ani = animation.FuncAnimation(
            self.fig, update, frames=steps, interval=interval, blit=True, repeat=False
        )




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

    #     Parameters
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