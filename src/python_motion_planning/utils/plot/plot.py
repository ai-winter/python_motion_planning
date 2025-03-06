"""
Plot tools 2D
@author: huiming zhou
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ..environment.env import Env, Grid, Map, Node


class Plot:
    def __init__(self, start, goal, env: Env):
        self.start = Node(start, start, 0, 0)
        self.goal = Node(goal, goal, 0, 0)
        self.env = env
        self.fig = plt.figure("planning")
        self.ax = self.fig.add_subplot()

    def animation(self, path: list, name: str, cost: float = None, expand: list = None, history_pose: list = None,
                  predict_path: list = None, lookahead_pts: list = None, cost_curve: list = None,
                  ellipse: np.ndarray = None) -> None:
        name = name + "\ncost: " + str(cost) if cost else name
        self.plotEnv(name)
        if expand is not None:
            self.plotExpand(expand)
        if history_pose is not None:
            self.plotHistoryPose(history_pose, predict_path, lookahead_pts)
        if path is not None:
            self.plotPath(path)

        if cost_curve:
            plt.figure("cost curve")
            self.plotCostCurve(cost_curve, name)

        if ellipse is not None:
            self.plotEllipse(ellipse)

        plt.show()

    def plotEnv(self, name: str) -> None:
        '''
        Plot environment with static obstacles.

        Parameters
        ----------
        name: Algorithm name or some other information
        '''
        plt.plot(self.start.x, self.start.y, marker="s", color="#ff0000")
        plt.plot(self.goal.x, self.goal.y, marker="s", color="#1155cc")

        if isinstance(self.env, Grid):
            obs_x = [x[0] for x in self.env.obstacles]
            obs_y = [x[1] for x in self.env.obstacles]
            plt.plot(obs_x, obs_y, "sk")

        if isinstance(self.env, Map):
            ax = self.fig.add_subplot()
            # boundary
            for (ox, oy, w, h) in self.env.boundary:
                ax.add_patch(patches.Rectangle(
                        (ox, oy), w, h,
                        edgecolor='black',
                        facecolor='black',
                        fill=True
                    )
                )
            # rectangle obstacles
            for (ox, oy, w, h) in self.env.obs_rect:
                ax.add_patch(patches.Rectangle(
                        (ox, oy), w, h,
                        edgecolor='black',
                        facecolor='gray',
                        fill=True
                    )
                )
            # circle obstacles
            for (ox, oy, r) in self.env.obs_circ:
                ax.add_patch(patches.Circle(
                        (ox, oy), r,
                        edgecolor='black',
                        facecolor='gray',
                        fill=True
                    )
                )

        plt.title(name)
        plt.axis("equal")

    def plotExpand(self, expand: list) -> None:
        '''
        Plot expanded grids using in graph searching.

        Parameters
        ----------
        expand: Expanded grids during searching
        '''
        if self.start in expand:
            expand.remove(self.start)
        if self.goal in expand:
            expand.remove(self.goal)

        count = 0
        if isinstance(self.env, Grid):
            for x in expand:
                count += 1
                plt.plot(x.x, x.y, color="#dddddd", marker='s')
                plt.gcf().canvas.mpl_connect('key_release_event',
                                            lambda event: [exit(0) if event.key == 'escape' else None])
                if count < len(expand) / 3:         length = 20
                elif count < len(expand) * 2 / 3:   length = 30
                else:                               length = 40
                if count % length == 0:             plt.pause(0.001)
        
        if isinstance(self.env, Map):
            for x in expand:
                count += 1
                if x.parent:
                    plt.plot([x.parent[0], x.x], [x.parent[1], x.y], 
                        color="#dddddd", linestyle="-")
                    plt.gcf().canvas.mpl_connect('key_release_event',
                                                 lambda event:
                                                 [exit(0) if event.key == 'escape' else None])
                    if count % 10 == 0:
                        plt.pause(0.001)

        plt.pause(0.01)

    def plotPath(self, path: list, path_color: str='#13ae00', path_style: str="-") -> None:
        '''
        Plot path in global planning.

        Parameters
        ----------
        path: Path found in global planning
        '''
        path_x = [path[i][0] for i in range(len(path))]
        path_y = [path[i][1] for i in range(len(path))]
        plt.plot(path_x, path_y, path_style, linewidth='2', color=path_color)
        plt.plot(self.start.x, self.start.y, marker="s", color="#ff0000")
        plt.plot(self.goal.x, self.goal.y, marker="s", color="#1155cc")

    def plotAgent(self, pose: tuple, radius: float=1) -> None:
        '''
        Plot agent with specifical pose.

        Parameters
        ----------
        pose: Pose of agent
        radius: Radius of agent
        '''
        x, y, theta = pose
        ref_vec = np.array([[radius / 2], [0]])
        rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta),  np.cos(theta)]])
        end_pt = rot_mat @ ref_vec + np.array([[x], [y]])

        try:
            self.ax.artists.pop()
            for art in self.ax.get_children():
                if isinstance(art, matplotlib.patches.FancyArrow):
                    art.remove()
        except:
            pass

        self.ax.arrow(x, y, float(end_pt[0]) - x, float(end_pt[1]) - y,
                width=0.1, head_width=0.40, color="r")
        circle = plt.Circle((x, y), radius, color="r", fill=False)
        self.ax.add_artist(circle)

    def plotHistoryPose(self, history_pose, predict_path=None, lookahead_pts=None) -> None:
        lookahead_handler = None
        for i, pose in enumerate(history_pose):
            if i < len(history_pose) - 1:
                plt.plot([history_pose[i][0], history_pose[i + 1][0]],
                    [history_pose[i][1], history_pose[i + 1][1]], c="#13ae00")
                if predict_path is not None:
                    plt.plot(predict_path[i][:, 0], predict_path[i][:, 1], c="#ddd")
            i += 1

            # agent
            self.plotAgent(pose)

            # lookahead
            if lookahead_handler is not None:
                lookahead_handler.remove()
            if lookahead_pts is not None:
                try:
                    lookahead_handler = self.ax.scatter(lookahead_pts[i][0], lookahead_pts[i][1], c="b")
                except:
                    lookahead_handler = self.ax.scatter(lookahead_pts[-1][0], lookahead_pts[-1][1], c="b")

            plt.gcf().canvas.mpl_connect('key_release_event',
                                        lambda event: [exit(0) if event.key == 'escape' else None])
            if i % 5 == 0:             plt.pause(0.03)

    def plotCostCurve(self, cost_list: list, name: str) -> None:
        '''
        Plot cost curve with epochs using in evolutionary searching.

        Parameters
        ----------
        cost_list: Cost with epochs
        name: Algorithm name or some other information
        '''
        plt.plot(cost_list, color="b")
        plt.xlabel("epochs")
        plt.ylabel("cost value")
        plt.title(name)
        plt.grid()

    def plotEllipse(self, ellipse: np.ndarray, color: str = 'darkorange', linestyle: str = '--', linewidth: float = 2):
        plt.plot(ellipse[0, :], ellipse[1, :], linestyle=linestyle, color=color, linewidth=linewidth)

    def connect(self, name: str, func) -> None:
        self.fig.canvas.mpl_connect(name, func)

    def clean(self):
        plt.cla()

    def update(self):
        self.fig.canvas.draw_idle()

    @staticmethod
    def plotArrow(x, y, theta, length, color):
        angle = np.deg2rad(30)
        d = 0.5 * length
        w = 2

        x_start, y_start = x, y
        x_end = x + length * np.cos(theta)
        y_end = y + length * np.sin(theta)

        theta_hat_L = theta + np.pi - angle
        theta_hat_R = theta + np.pi + angle

        x_hat_start = x_end
        x_hat_end_L = x_hat_start + d * np.cos(theta_hat_L)
        x_hat_end_R = x_hat_start + d * np.cos(theta_hat_R)

        y_hat_start = y_end
        y_hat_end_L = y_hat_start + d * np.sin(theta_hat_L)
        y_hat_end_R = y_hat_start + d * np.sin(theta_hat_R)

        plt.plot([x_start, x_end], [y_start, y_end], color=color, linewidth=w)
        plt.plot([x_hat_start, x_hat_end_L], [y_hat_start, y_hat_end_L], color=color, linewidth=w)
        plt.plot([x_hat_start, x_hat_end_R], [y_hat_start, y_hat_end_R], color=color, linewidth=w)

    @staticmethod
    def plotCar(x, y, theta, width, length, color):
        theta_B = np.pi + theta

        xB = x + length / 4 * np.cos(theta_B)
        yB = y + length / 4 * np.sin(theta_B)

        theta_BL = theta_B + np.pi / 2
        theta_BR = theta_B - np.pi / 2

        x_BL = xB + width / 2 * np.cos(theta_BL)        # Bottom-Left vertex
        y_BL = yB + width / 2 * np.sin(theta_BL)
        x_BR = xB + width / 2 * np.cos(theta_BR)        # Bottom-Right vertex
        y_BR = yB + width / 2 * np.sin(theta_BR)

        x_FL = x_BL + length * np.cos(theta)               # Front-Left vertex
        y_FL = y_BL + length * np.sin(theta)
        x_FR = x_BR + length * np.cos(theta)               # Front-Right vertex
        y_FR = y_BR + length * np.sin(theta)

        plt.plot([x_BL, x_BR, x_FR, x_FL, x_BL],
                 [y_BL, y_BR, y_FR, y_FL, y_BL],
                 linewidth=1, color=color)

        Plot.plotArrow(x, y, theta, length / 2, color)