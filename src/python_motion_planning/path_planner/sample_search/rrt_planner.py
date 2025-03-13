"""
@file: rrt_planner.py
@breif: RRT path planning
@author: Winter
@update: 2023.1.17
"""
import math
import numpy as np

from typing import List, Tuple, Dict

from python_motion_planning.path_planner import PathPlanner
from python_motion_planning.common.structure import Node, Env
from python_motion_planning.common.geometry import Point3d

class RRTPlanner(PathPlanner):
    """
    Class for RRT path planning.

    Parameters:
        env (Env): environment object
        params (dict): parameters

    References:
        [1] Rapidly-Exploring Random Trees: A New Tool for Path Planning
    """
    def __init__(self, env: Env, params: dict) -> None:
        super().__init__(env, params)

    def __str__(self) -> str:
        return "Rapidly-exploring Random Tree(RRT)"

    def plan(self, start: Point3d, goal: Point3d) -> Tuple[List[Point3d], List[Dict]]:
        """
        Rapidly-exploring Random Tree motion plan function.

        Parameters:
            start (Point3d): The starting point of the planning path.
            goal (Point3d): The goal point of the planning path.

        Returns:
            path (List[Point3d]): The planned path from start to goal.
            visual_info (List[Dict]): Information for visualization
        """
        self.start = Node(start, start, 0, 0)
        self.goal = Node(goal, goal, 0, 0)

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
                if dist <= self.max_dist and not self.collision_checker(node_new.current, self.goal.current):
                    self.goal.parent = node_new.current
                    self.goal.g = node_new.g + self.dist(self.goal, node_new)
                    sample_list[self.goal.current] = self.goal
                    cost, path = self.extractPath(sample_list)
                    return path, [
                        {"type": "value", "data": True, "name": "success"},
                        {"type": "value", "data": cost, "name": "cost"},
                        
                        {"type": "path", "name": "line", "data": [
                            [[n.parent.x(), n.x], [n.parent.y(), n.y]] for n in sample_list.values()
                            if n.parent is not None
                        ], "props": {"color": "#dddddd"}},
                        {"type": "path", "data": path, "name": "normal"}
                    ]
                        
        return path, [
            {"type": "value", "data": False, "name": "success"},
            {"type": "value", "data": 0, "name": "cost"},
            
            {"type": "path", "data": [], "name": "normal"},
            {"type": "path", "name": "line", "data": []}
        ]

    def generateRandomNode(self) -> Node:
        """
        Generate a random node to extend exploring tree.

        Returns:
            node (Node): a random node based on sampling
        """
        if np.random.random() > self.goal_sample_rate:
            return Node(Point3d(
                    np.random.uniform(self.delta, self.env.x_range - self.delta),
                    np.random.uniform(self.delta, self.env.y_range - self.delta)
                ), None, 0, 0
            )
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
        node_new = Node(
            Point3d(node_near.x + dist * math.cos(theta),
                    node_near.y + dist * math.sin(theta)
            ), node_near.current, node_near.g + dist, 0
        )
        
        # obstacle check
        if self.collision_checker(node_new.current, node_near.current):
            return None
        return node_new

    def extractPath(self, closed_set):
        """
        Extract the path based on the CLOSED set.

        Parameters:
            closed_set (list): CLOSED set

        Returns
            cost (float): the cost of planning path
            path (list): the planning path
        """
        node = closed_set[self.goal.current]
        path = [node.current]
        cost = node.g
        while node != self.start:
            node_parent = closed_set[node.parent]
            node = node_parent
            path.append(node.current)
        path = list({}.fromkeys(path).keys())
        return cost, list(reversed(path))
