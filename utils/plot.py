"""
Plot tools 2D
@author: huiming zhou
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .env import Env, Grid, Map, Node


class Plot:
    def __init__(self, start, goal, env: Env):
        self.start = Node(start, start, 0, 0)
        self.goal = Node(goal, goal, 0, 0)
        self.env = env
        self.fig = plt.figure("planning")
        self.ax = self.fig.add_subplot()

    def animation(self, path, name, cost=None, expand=None, history_pose=None, cost_curve=None) -> None:
        name = name + "\ncost: " + str(cost) if cost else name
        self.plotEnv(name)
        if expand:
            self.plotExpand(expand)
        if history_pose:
            self.plotHistoryPose(history_pose)
        self.plotPath(path)

        if cost_curve:
            plt.figure("cost curve")
            self.plotCostCurve(cost_curve, name)

        plt.show()

    def plotEnv(self, name: str) -> None:
        '''
        Plot environment with static obstacles.

        Parameters
        ----------
        name: Algorithm name or some other information
        '''
        plt.plot(self.start.current[0], self.start.current[1], marker="s", color="#ff0000")
        plt.plot(self.goal.current[0], self.goal.current[1], marker="s", color="#1155cc")

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
                plt.plot(x.current[0], x.current[1], color="#dddddd", marker='s')
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
                    plt.plot([x.parent[0], x.current[0]], [x.parent[1], x.current[1]], 
                        color="#dddddd", linestyle="-")
                    plt.gcf().canvas.mpl_connect('key_release_event',
                                                 lambda event:
                                                 [exit(0) if event.key == 'escape' else None])
                    if count % 10 == 0:
                        plt.pause(0.001)

        plt.pause(0.01)

    def plotPath(self, path: list) -> None:
        '''
        Plot path in global planning.

        Parameters
        ----------
        path: Path found in global planning
        '''
        path_x = [path[i][0] for i in range(len(path))]
        path_y = [path[i][1] for i in range(len(path))]
        plt.plot(path_x, path_y, linewidth='2', color='#13ae00')
        plt.plot(self.start.current[0], self.start.current[1], marker="s", color="#ff0000")
        plt.plot(self.goal.current[0], self.goal.current[1], marker="s", color="#1155cc")

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

    def plotHistoryPose(self, history_pose) -> None:
        count = 0
        for pose in history_pose:
            if count < len(history_pose) - 1:
                plt.plot([history_pose[count][0], history_pose[count + 1][0]],
                    [history_pose[count][1], history_pose[count + 1][1]], c="r")
            count += 1
            self.plotAgent(pose)
            plt.gcf().canvas.mpl_connect('key_release_event',
                                        lambda event: [exit(0) if event.key == 'escape' else None])
            if count < len(history_pose) / 3:         length = 5
            elif count < len(history_pose) * 2 / 3:   length = 10
            else:                                     length = 20
            if count % length == 0:             plt.pause(0.01)

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

    def connect(self, name: str, func) -> None:
        self.fig.canvas.mpl_connect(name, func)

    def clean(self):
        plt.cla()

    def update(self):
        self.fig.canvas.draw_idle()

