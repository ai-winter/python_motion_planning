"""
@file: visualization.py
@breif: visualization
@author: Yang Haodong, Wu Maojia
@update: 2025.3.29
"""
from typing import Union
from collections import namedtuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from python_motion_planning.common.env import TYPES, world, Grid

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
    def __init__(self, fig_name: str):
        self.fig = plt.figure(fig_name)
        self.ax = self.fig.add_subplot()

        # colors
        self.cmap_dict = {
            TYPES.FREE: "#ffffff",
            TYPES.OBSTACLE: "#000000",
            TYPES.START: "#ff0000",
            TYPES.GOAL: "#1155cc",
            TYPES.CUSTOM: "#dddddd",
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
            print(grid_map.type_map.array)
            plt.imshow(grid_map.type_map.array, cmap=self.cmap, norm=self.norm, origin='lower', interpolation='nearest')
            if equal: 
                plt.axis("equal")

        elif grid_map.ndim == 3:
            raise NotImplementedError
        else:
            raise NotImplementedError

        # # self.grid_map = env.grid_map
        # self.grid_map = np.zeros((env.y_range, env.x_range))
        # for (ox, oy) in env.getNodes():
        #     self.grid_map[oy, ox] = TYPES.OBSTACLE_GRID
        
        # plt.imshow(self.map, cmap=self.cmap, norm=self.norm, origin='lower', interpolation='nearest')
        # plt.axis("equal")

    # def plotGrids(self, grids: list) -> None:
    #     '''
    #     Plot grids in grid map.

    #     Parameters
    #     ----------
    #     grid_dict: grid information
    #     '''  
    #     if self.grid_map is None:
    #         raise RuntimeWarning("Grid map is Null")
        
        
    #     for grid in grids:
    #         grid_x = int(grid["x"])
    #         grid_y = int(grid["y"])
    #         grid_name = grid["name"]
    #         self.grid_map[grid_y, grid_x] = self.cmap_dict[grid_name].idx

    #     plt.imshow(self.grid_map, cmap=self.cmap, norm=self.norm, origin='lower', interpolation='nearest')

    def setTitle(self, title: str) -> None:
        plt.title(title)

    # def plotPath(self, path: list, axis_equal: bool=True, name: str="normal", props: dict={}) -> None:
    #     '''
    #     Plot path-like information.
    #     '''
    #     path_style = props["style"] if "style" in props.keys() else "-"
    #     path_color = props["color"] if "color" in props.keys() else "#13ae00"
    #     path_label = props["label"] if "label" in props.keys() else None
    #     linewidth = props["width"] if "width" in props.keys() else 2
    #     marker = props["marker"] if "marker" in props.keys() else None

    #     if name == "normal":
    #         path_x = [path[i][0] for i in range(len(path))]
    #         path_y = [path[i][1] for i in range(len(path))]
    #         plt.plot(path_x, path_y, path_style, lw=linewidth, color=path_color, label=path_label, marker=marker)
    #     elif name == "line":
    #         for line in path:
    #             plt.plot(line[0], line[1], path_style, lw=linewidth, color=path_color, label=path_label, marker=marker)
    #     else:
    #         raise NotImplementedError
        
    #     if axis_equal:
    #         plt.axis("equal")
        
    #     if path_label:
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