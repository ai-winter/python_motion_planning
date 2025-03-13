"""
@file: jps_planner.py
@breif: Jump Point Search path planning
@author: Yang Haodong
@update: 2024.2.11
"""
import heapq

from typing import List, Tuple, Dict

from .astar_planner import AStarPlanner

from python_motion_planning.common.utils import LOG
from python_motion_planning.common.structure import Node, Env
from python_motion_planning.common.geometry import Point3d

class JPSPlanner(AStarPlanner):
    """
    Class for JPS path planning.

    Parameters:
        env (Env): environment object
        params (dict): parameters

    References:
        [1] Online Graph Pruning for Pathfinding On Grid Maps
    """
    def __init__(self, env: Env, params: dict) -> None:
        super().__init__(env, params)
    
    def __str__(self) -> str:
        return "Jump Point Search(JPS)"

    def plan(self, start: Point3d, goal: Point3d) -> Tuple[List[Point3d], List[Dict]]:
        """
        Jump point search motion plan function.

        Parameters:
            start (Point3d): The starting point of the planning path.
            goal (Point3d): The goal point of the planning path.

        Returns:
            path (List[Point3d]): The planned path from start to goal.
            visual_info (List[Dict]): Information for visualization
        """
        self.start = Node(start, start, 0, 0)
        self.goal = Node(goal, goal, 0, 0)

        # OPEN set with priority and CLOSED set
        OPEN = []
        heapq.heappush(OPEN, self.start)
        CLOSED = dict()

        while OPEN:
            node = heapq.heappop(OPEN)

            # exists in CLOSED set
            if node.current in CLOSED:
                continue

            # goal found
            if node == self.goal:
                CLOSED[self.goal.current] = node
                cost, path = self.extractPath(CLOSED)
                LOG.INFO(f"{str(self)} PathPlanner Planning Successfully. Cost: {cost}")
                return path, [
                    {"type": "value", "data": True, "name": "success"},
                    {"type": "value", "data": cost, "name": "cost"},
                    {"type": "path", "data": path, "name": "normal"},
                    {"type": "grids", "data": [n.current for n in CLOSED.values()], "name": "expand"}
                ]

            jp_list = []
            for motion in self.motions:
                jp = self.jump(node, motion)
                # exists and not in CLOSED set
                if jp and jp.current not in CLOSED:
                    jp.parent = node.current
                    jp.h = self.h(jp, self.goal)
                    jp_list.append(jp)

            for jp in jp_list:
                # update OPEN set
                heapq.heappush(OPEN, jp)

                # goal found
                if jp == self.goal:
                    break
            
            CLOSED[node.current] = node
        

        LOG.INFO("Planning Failed.")
        return path, [
            {"type": "value", "data": False, "name": "success"},
            {"type": "value", "data": 0.0, "name": "cost"},
            {"type": "path", "data": [], "name": "normal"},
            {"type": "grids", "data": [], "name": "expand"}
        ]

    def jump(self, node: Node, motion: Node):
        """
        Jumping search recursively.

        Parameters:
            node (Node): current node
            motion (Node): the motion that current node executes

        Returns:
            jump_point (Node): jump point or None if searching fails
        """
        # explore a new node
        new_node = node + motion
        new_node.parent = node.current
        new_node.h = self.h(new_node, self.goal)

        # hit the obstacle
        if (new_node.x, new_node.y) in self.obstacles:
            return None

        # goal found
        if new_node == self.goal:
            return new_node

        # diagonal
        if motion.x and motion.y:
            # if exists jump point at horizontal or vertical
            x_dir = Node(Point3d(motion.x, 0, 0), None, 1, None)
            y_dir = Node(Point3d(0, motion.y, 0), None, 1, None)
            if self.jump(new_node, x_dir) or self.jump(new_node, y_dir):
                return new_node
            
        # if exists forced neighbor
        if self.detectForceNeighbor(new_node, motion):
            return new_node
        else:
            return self.jump(new_node, motion)

    def detectForceNeighbor(self, node, motion):
        """
        Detect forced neighbor of node.

        Parameters:
            node (Node): current node
            motion (Node): the motion that current node executes

        Returns:
            flag (bool): True if current node has forced neighbor else Flase
        """
        x, y = node.x, node.y
        x_dir, y_dir = motion.x, motion.y

        # horizontal
        if x_dir and not y_dir:
            if (x, y + 1) in self.obstacles and \
                (x + x_dir, y + 1) not in self.obstacles:
                return True
            if (x, y - 1) in self.obstacles and \
                (x + x_dir, y - 1) not in self.obstacles:
                return True
        
        # vertical
        if not x_dir and y_dir:
            if (x + 1, y) in self.obstacles and \
                (x + 1, y + y_dir) not in self.obstacles:
                return True
            if (x - 1, y) in self.obstacles and \
                (x - 1, y + y_dir) not in self.obstacles:
                return True
        
        # diagonal
        if x_dir and y_dir:
            if (x - x_dir, y) in self.obstacles and \
                (x - x_dir, y + y_dir) not in self.obstacles:
                return True
            if (x, y - y_dir) in self.obstacles and \
                (x + x_dir, y - y_dir) not in self.obstacles:
                return True
        
        return False

