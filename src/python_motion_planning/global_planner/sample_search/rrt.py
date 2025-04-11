"""
@file: rrt.py
@breif: RRT motion planning
@author: Yang Haodong, Wu Maojia
@update: 2024.6.23
"""
import math
import numpy as np

from .sample_search import SampleSearcher
from python_motion_planning.utils import Env, Node, Map


class RRT(SampleSearcher):
    """
    Class for RRT motion planning.

    Parameters:
        start (tuple): start point coordinate
        goal (tuple): goal point coordinate
        env (Map): environment
        max_dist (float): Maximum expansion distance one step
        sample_num (int): Maximum number of sample points
        goal_sample_rate (float): heuristic sample

    Examples:
        >>> import python_motion_planning as pmp
        >>> planner = pmp.RRT((5, 5), (45, 25), pmp.Map(51, 31))
        >>> cost, path, expand = planner.plan()     # planning results only
        >>> planner.plot.animation(path, str(planner), cost, expand)  # animation
        >>> planner.run()       # run both planning and animation

    References:
        [1] Rapidly-Exploring Random Trees: A New Tool for Path Planning
    """
    def __init__(self, start: tuple, goal: tuple, env: Map, max_dist: float = 0.5,
        sample_num: int = 10000, goal_sample_rate: float = 0.05) -> None:
        super().__init__(start, goal, env)
        # Maximum expansion distance one step
        self.max_dist = max_dist
        # Maximum number of sample points
        self.sample_num = sample_num
        # heuristic sample
        self.goal_sample_rate = goal_sample_rate

    def __str__(self) -> str:
        return "Rapidly-exploring Random Tree(RRT)"

    def plan(self) -> tuple:
        """
        RRT motion plan function.

        Returns:
            cost (float): path cost
            path (list): planning path
            expand (list): expanded (sampled) nodes list
        """
        # Sampled list
        sample_list = {self.start.current: self.start}

        # main loop
        for _ in range(self.sample_num):
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
                    return cost, path, list(sample_list.values())

        return 0, None, list(sample_list.values())

    def run(self) -> None:
        """
        Running both plannig and animation.
        """
        cost, path, expand = self.plan()
        self.plot.animation(path, str(self), cost, expand)

    def generateRandomNode(self) -> Node:
        """
        Generate a random node to extend exploring tree.

        Returns:
            node (Node): a random node based on sampling
        """
        if np.random.random() > self.goal_sample_rate:
            current = (np.random.uniform(self.delta, self.env.x_range - self.delta),
                    np.random.uniform(self.delta, self.env.y_range - self.delta))
            return Node(current, None, 0, 0)
        return self.goal

    def getNearest(self, node_list: list, node: Node) -> Node:
        """
        Get the node from `node_list` that is nearest to `node`.

        Parameters:
            node_list (list): exploring list
            node (Node): currently generated node

        Returns:
            node (Node): nearest node
        """
        # find nearest neighbor
        dist = [self.dist(node, nd) for nd in node_list]
        node_near = node_list[int(np.argmin(dist))]

        # regular and generate new node
        dist, theta = self.dist(node_near, node), self.angle(node_near, node)
        dist = min(self.max_dist, dist)
        node_new = Node((node_near.x + dist * math.cos(theta),
                        (node_near.y + dist * math.sin(theta))),
                         node_near.current, node_near.g + dist, 0)
        
        # obstacle check
        if self.isCollision(node_new, node_near):
            return None
        return node_new

    def extractPath(self, closed_list: dict) -> tuple:
        """
        Extract the path based on the CLOSED list.

        Parameters:
            closed_list (dict): CLOSED list

        Returns
            cost (float): the cost of planning path
            path (list): the planning path
        """
        node = closed_list[self.goal.current]
        path = [node.current]
        cost = node.g
        while node != self.start:
            node_parent = closed_list[node.parent]
            node = node_parent
            path.append(node.current)

        return cost, path
