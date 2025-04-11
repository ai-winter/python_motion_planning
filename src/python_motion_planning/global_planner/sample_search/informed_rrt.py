"""
@file: informed_rrt.py
@breif: Informed RRT* motion planning
@author: Yang Haodong, Wu Maojia
@update: 2024.6.23
"""
import numpy as np
from functools import partial
import matplotlib.pyplot as plt

from .rrt_star import RRTStar
from python_motion_planning.utils import Env, Node, Map


class Ellipse:
    """
    Ellipse sampling.
    """
    @staticmethod
    def transform(a: float, c: float, p1: tuple, p2: tuple) -> np.ndarray:
        # center
        center_x = (p1[0] + p2[0]) / 2
        center_y = (p1[1] + p2[1]) / 2

        # rotation
        theta = - np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

        # transform
        b = np.sqrt(a ** 2 - c ** 2)
        T = np.array([[ a * np.cos(theta), b * np.sin(theta), center_x],
                      [-a * np.sin(theta), b * np.cos(theta), center_y],
                      [                 0,                 0,        1]])
        return T


class InformedRRT(RRTStar):
    """
    Class for Informed RRT* motion planning.

    Parameters:
        start (tuple): start point coordinate
        goal (tuple): goal point coordinate
        env (Map): environment
        max_dist (float): Maximum expansion distance one step
        sample_num (int): Maximum number of sample points
        r (float): optimization radius
        goal_sample_rate (float): heuristic sample

    Examples:
        >>> import python_motion_planning as pmp
        >>> planner = pmp.InformedRRT((5, 5), (45, 25), pmp.Map(51, 31))
        >>> cost, path, expand = planner.plan()     # planning results only
        >>> planner.plot.animation(path, str(planner), cost, expand)  # animation
        >>> planner.run()       # run both planning and animation

    References:
        [1] Optimal Sampling-based Path Planning Focused via Direct Sampling of an Admissible Ellipsoidal heuristic
    """
    def __init__(self, start: tuple, goal: tuple, env: Map, max_dist: float = 0.5,
                 sample_num: int = 1500, r: float = 12.0, goal_sample_rate: float = 0.05) -> None:
        super().__init__(start, goal, env, max_dist, sample_num, goal_sample_rate)
        # optimization radius
        self.r = r
        # best planning cost
        self.c_best = float("inf")
        # distance between start and goal
        self.c_min = self.dist(self.start, self.goal)
        # ellipse sampling
        self.transform = partial(Ellipse.transform, c=self.c_min / 2, p1=start, p2=goal)
    
    def __str__(self) -> str:
        return "Informed RRT*"

    def plan(self) -> tuple:
        """
        Informed-RRT* motion plan function.

        Returns:
            cost (float): path cost
            path (list): planning path
            expand (list): expanded (sampled) nodes list
        """
        # Sampled list
        sample_list = {self.start.current: self.start}

        best_cost, best_path = float("inf"), None

        # main loop
        for i in range(self.sample_num):
            # generate a random node in the map
            node_rand = self.generateRandomNode()

            # visited
            if node_rand.current in sample_list:
                continue

            # generate new node
            node_new = self.getNearest(list(sample_list.values()), node_rand)
            if node_new:
                sample_list[node_new.current] = node_new
                dist = self.dist(node_new, self.goal)
                # goal found
                if dist <= self.max_dist and not self.isCollision(node_new, self.goal):
                    self.goal.parent = node_new.current
                    self.goal.g = node_new.g + self.dist(self.goal, node_new)
                    sample_list[self.goal.current] = self.goal
                    cost, path = self.extractPath(sample_list)
                    if path and cost < best_cost:
                        best_cost, best_path = cost, path
                        self.c_best = best_cost

        return best_cost, best_path, list(sample_list.values())

    def run(self) -> None:
        """
        Running both plannig and animation.
        """
        cost, path, expand = self.plan()
        t = np.arange(0, 2 * np.pi + 0.1, 0.1)
        x = [np.cos(it) for it in t]
        y = [np.sin(it) for it in t]
        z = [1 for _ in t]
        fx = self.transform(self.c_best / 2) @ np.array([x, y, z])
        fx[0, :] += x
        fx[1, :] += y
        self.plot.animation(path, str(self), cost, expand, ellipse=fx)

    def generateRandomNode(self) -> Node:
        """
        Generate a random node to extend exploring tree.

        Returns:
            node (Node): a random node based on sampling
        """
        # ellipse sample
        if self.c_best < float("inf"):
            while True:
                # unit ball sample
                p = np.array([.0, .0, 1.])
                while True:
                    x, y = np.random.uniform(-1, 1), np.random.uniform(-1, 1)
                    if x ** 2 + y ** 2 < 1:
                        p[0], p[1] = x, y
                        break
                # transform to ellipse
                p_star = self.transform(self.c_best / 2) @ p.T
                if self.delta <= p_star[0] <= self.env.x_range - self.delta and \
                   self.delta <= p_star[1] <= self.env.y_range - self.delta:
                    return Node((p_star[0], p_star[1]), None, 0, 0)
        # random sample
        else:
            return super().generateRandomNode()