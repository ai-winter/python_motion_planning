"""
@file: rrt_connected.py
@breif: RRT-Connected motion planning
@author: Yang Haodong, Wu Maojia
@update: 2024.6.23
"""
import math

from .rrt import RRT
from python_motion_planning.utils import Env, Node, Map


class RRTConnect(RRT):
    """
    Class for RRT-Connect motion planning.

    Parameters:
        start (tuple): start point coordinate
        goal (tuple): goal point coordinate
        env (Map): environment
        max_dist (float): Maximum expansion distance one step
        sample_num (int): Maximum number of sample points
        goal_sample_rate (float): heuristic sample

    Examples:
        >>> import python_motion_planning as pmp
        >>> planner = pmp.RRTConnect((5, 5), (45, 25), pmp.Map(51, 31))
        >>> cost, path, expand = planner.plan()     # planning results only
        >>> planner.plot.animation(path, str(planner), cost, expand)  # animation
        >>> planner.run()       # run both planning and animation

    References:
        [1] RRT-Connect: An Efficient Approach to Single-Query Path Planning
    """
    def __init__(self, start: tuple, goal: tuple, env: Map, max_dist: float = 0.5,
        sample_num: int = 10000, goal_sample_rate: float = 0.05) -> None:
        super().__init__(start, goal, env, max_dist, sample_num, goal_sample_rate)
    
    def __str__(self) -> str:
        return "RRT-Connect"

    def plan(self) -> tuple:
        """
        RRT-Connected motion plan function.

        Returns:
            cost (float): path cost
            path (list): planning path
            expand (list): expanded (sampled) nodes list
        """
        # Sampled list forward
        sample_list_f = {self.start.current: self.start}
        # Sampled list backward
        sample_list_b = {self.goal.current: self.goal}

        for _ in range(self.sample_num):
            # generate a random node in the map
            node_rand = self.generateRandomNode()            
            # generate new node
            node_new = self.getNearest(list(sample_list_f.values()), node_rand)
            if node_new:
                sample_list_f[node_new.current] = node_new
                # backward exploring
                node_new_b = self.getNearest(list(sample_list_b.values()), node_new)
                if node_new_b:
                    sample_list_b[node_new_b.current] = node_new_b

                    # greedy extending
                    while True:
                        if node_new_b == node_new:
                            cost, path = self.extractPath(node_new, sample_list_b, sample_list_f)
                            expand = self.getExpand(list(sample_list_b.values()), list(sample_list_f.values()))
                            return cost, path, expand

                        dist = min(self.max_dist, self.dist(node_new, node_new_b))
                        theta = self.angle(node_new_b, node_new)
                        node_new_b2 = Node((node_new_b.x + dist * math.cos(theta),
                                           (node_new_b.y + dist * math.sin(theta))),
                                            node_new_b.current, node_new_b.g + dist, 0)

                        if not self.isCollision(node_new_b2, node_new_b):
                            sample_list_b[node_new_b2.current] = node_new_b2
                            node_new_b = node_new_b2
                        else:
                            break

            if len(sample_list_b) < len(sample_list_f):
                sample_list_f, sample_list_b = sample_list_b, sample_list_f

        return 0, None, None

    def extractPath(self, boundary: Node, sample_list_b: dict, sample_list_f: dict) -> tuple:
        """
        Extract the path based on the CLOSED set.

        Parameters:
            boundary (Node): the boundary node
            sample_list_b (dict): Sample list backward
            sample_list_f (dict): Sample list forward

        Returns:
            cost (float): the cost of planning path
            path (list): the planning path
        """
        if self.start.current in sample_list_b:
            sample_list_f, sample_list_b = sample_list_b, sample_list_f

        # forward
        node = sample_list_f[boundary.current]
        path_f = [node.current]
        cost = node.g
        while node != self.start:
            node_parent = sample_list_f[node.parent]
            node = node_parent
            path_f.append(node.current)

        # backward
        node = sample_list_b[boundary.current]
        path_b = []
        cost += node.g
        while node != self.goal:
            node_parent = sample_list_b[node.parent]
            node = node_parent
            path_b.append(node.current)        

        return cost, list(reversed(path_f)) + path_b

    def getExpand(self, sample_list_b: list, sample_list_f: list) -> list:
        """
        Get the expand list from sample list.

        Parameters:
            sample_list_b (list): Sample list backward
            sample_list_f (list): Sample list forward

        Returns:
            expand (list): expand list
        """
        expand = []
        tree_size = max(len(sample_list_f), len(sample_list_b))
        for k in range(tree_size):
            if k < len(sample_list_f):
                expand.append(sample_list_f[k])
            if k < len(sample_list_b):
                expand.append(sample_list_b[k])

        return expand