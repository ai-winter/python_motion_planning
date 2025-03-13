"""
@file: lpa_star_planner.py
@breif: Lifelong Planning A* path planning
@author: Yang Haodong
@update: 2024.2.11
"""
import math
import heapq

from typing import List, Tuple, Dict

from python_motion_planning.path_planner import PathPlanner
from python_motion_planning.common.utils import LOG
from python_motion_planning.common.structure import Node, Env
from python_motion_planning.common.utils import Visualizer
from python_motion_planning.common.geometry import Point3d

class LNode(Node):
    """
    Class for LPA* nodes.

    Parameters:
        current (tuple): current coordinate
        g (float): minimum cost moving from start(predict)
        rhs (float): minimum cost moving from start(value)
        key (list): priority
    """
    def __init__(self, current: Point3d, g: float, rhs: float, key: list) -> None:
        self.current = current
        self.g = g
        self.rhs = rhs
        self.key = key

    def __add__(self, node):
        return LNode(
            Point3d(self.current.x() + node.current.x(),
                    self.current.y() + node.current.y(),
                    self.current.theta() + node.current.theta(),
            ), self.g, self.rhs, self.key
        )

    def __eq__(self, node: 'Node') -> bool:
        return (self.x == node.x and self.y == node.y)

    def __lt__(self, node) -> bool:
        return self.key < node.key

    def __str__(self) -> str:
        return f"----------\ncurrent:({self.x}, {self.y})\ng:{self.g}\nrhs:{self.rhs}\nk:{self.key}----------"

class LPAStarPlanner(PathPlanner):
    """
    Class for LPA* motion planning.

    Parameters:
        env (Env): environment object
        params (dict): parameters

    References:
        [1] Lifelong Planning A*
    """
    def __init__(self, env: Env, params: dict) -> None:
        super().__init__(env, params)
        # allowed motions
        self.motions = [
            LNode(Point3d(-1, 0, 0), None, 1, None), LNode(Point3d(-1,  1, 0), None, math.sqrt(2), None),
            LNode(Point3d( 0, 1, 0), None, 1, None), LNode(Point3d( 1,  1, 0), None, math.sqrt(2), None),
            LNode(Point3d( 1, 0, 0), None, 1, None), LNode(Point3d( 1, -1, 0), None, math.sqrt(2), None),
            LNode(Point3d( 0,-1, 0), None, 1, None), LNode(Point3d(-1, -1, 0), None, math.sqrt(2), None)
        ]

        # OPEN set and expand zone
        self.U, self.EXPAND = [], []

        # intialize global information, record history infomation of map grids
        grid_coor = {Point3d(i, j, 0) for i in range(self.env.x_range) for j in range(self.env.y_range)}
        self.map = [LNode(s, float("inf"), float("inf"), None) for s in grid_coor]

        # updated path
        self.path = []

        # updated cost
        self.p_cost = None

        # visualizer
        self.visualizer = None

    def __str__(self) -> str:
        return "Lifelong Planning A*"

    def plan(self, start: Point3d, goal: Point3d) -> Tuple[List[Point3d], List[Dict]]:
        """
        LPA* motion plan function.

        Parameters:
            start (Point3d): The starting point of the planning path.
            goal (Point3d): The goal point of the planning path.

        Returns:
            path (List[Point3d]): The planned path from start to goal.
            visual_info (List[Dict]): Information for visualization
        """
        self.start = LNode(start, float('inf'), 0.0, None)
        self.goal = LNode(goal, float('inf'), float('inf'), None)
        self.map[self.map.index(self.goal)] = self.goal
        self.map[self.map.index(self.start)] = self.start
        # OPEN set with priority
        self.start.key = self.calculateKey(self.start)
        heapq.heappush(self.U, self.start)
        # dynamic planning
        return self.planOnce()

    def planOnce(self):
        """
        LPA* dynamic planning once
        """
        self.computeShortestPath()
        self.p_cost, self.path = self.extractPath()

        LOG.INFO(f"{str(self)} PathPlanner Planning Successfully. Cost: {self.p_cost}")
        return self.path, [
            {"type": "value", "data": True, "name": "success"},
            {"type": "value", "data": self.p_cost, "name": "cost"},
            {"type": "path", "data": self.path, "name": "normal"},
            {"type": "grids", "data": [n.current for n in self.EXPAND], "name": "expand"},
            {"type": "callback", "data": self.OnPress, "name": "visualization"}
        ]

    def setVisualizer(self, visualizer: Visualizer) -> None:
        """
        Set visualizer for dynamic visualization.

        Parameters:
            visualizer (Visualizer): visualizer handler
        """
        self.visualizer = visualizer

    def OnPress(self, event):
        """
        Mouse button callback function.
        """
        x, y = int(event.xdata), int(event.ydata)
        if x < 0 or x > self.env.x_range - 1 or y < 0 or y > self.env.y_range - 1:
            LOG.INFO(f"Please choose right area within X-[0, {self.env.x_range}] Y-[0, {self.env.y_range}] instead of ({x}, {y})")
        else:
            LOG.INFO(f"Change position: x = {x}, y = {y}]")
            self.EXPAND = []
            node_change = self.map[self.map.index(LNode(Point3d(x, y, 0), None, None, None))]

            if (x, y) not in self.obstacles:
                self.obstacles.append((x, y))
            else:
                self.obstacles.remove((x, y))
                self.updateVertex(node_change)
            
            self.env.update(self.obstacles)
            self.collision_checker.update(self.obstacles)

            for node_n in self.getNeighbor(node_change):
                self.updateVertex(node_n)

            _, output = self.planOnce()        
        
            LOG.INFO(f"{str(self)} PathPlanner Planning Successfully. Cost: {self.p_cost}")

            # animation
            cost = 0
            self.visualizer.clean()
            self.visualizer.plotGridMap(self.env)
            self.visualizer.plotGrids([
                {"x": self.start.x, "y": self.start.y, "name": "start"},
                {"x": self.goal.x, "y": self.goal.y, "name": "goal"},
            ])
            for info in output:
                if info["type"] == "value" and info["name"] == "cost":
                    cost = info["data"]
                if info["type"] == "path" and info["name"] == "normal":
                    self.visualizer.plotPath(info["data"])
                if info["type"] == "grids" and info["name"] == "expand":
                    self.visualizer.plotGrids([
                        {"x": pt.x(), "y": pt.y(), "name": "custom"} for pt in info["data"]
                        if ((pt != self.start.current) and (pt != self.goal.current))
                    ])
            self.visualizer.setTitle(f"{str(self)}\ncost: {cost}")
            self.visualizer.update()

    def computeShortestPath(self) -> None:
        """
        Perceived dynamic obstacle information to optimize global path.
        """
        while True:
            node = min(self.U, key=lambda node: node.key)
            if node.key >= self.calculateKey(self.goal) and \
                    self.goal.rhs == self.goal.g:
                break

            self.U.remove(node)
            self.EXPAND.append(node)

            # Locally over-consistent -> Locally consistent
            if node.g > node.rhs:
                node.g = node.rhs
            # Locally under-consistent -> Locally over-consistent
            else:
                node.g = float("inf")
                self.updateVertex(node)

            for node_n in self.getNeighbor(node):
                self.updateVertex(node_n)

    def updateVertex(self, node: LNode) -> None:
        """
        Update the status and the current cost to node and it's neighbor.
        """
        # greed correction
        if node != self.start:
            node.rhs = min([node_n.g + self.cost(node_n, node)
                        for node_n in self.getNeighbor(node)])

        if node in self.U:
            self.U.remove(node)

        # Locally unconsistent nodes should be added into OPEN set (set U)
        if node.g != node.rhs:
            node.key = self.calculateKey(node)
            heapq.heappush(self.U, node)

    def calculateKey(self, node: LNode) -> list:
        """
        Calculate priority of node.
        """
        return [min(node.g, node.rhs) + self.h(node, self.goal),
                min(node.g, node.rhs)]

    def getNeighbor(self, node: LNode) -> list:
        """
        Find neighbors of node.

        Parameters
        ----------
        node: DNode
            current node

        Return
        ----------
        neighbors: list
            neighbors of current node
        """
        neighbors = []
        for motion in self.motions:
            n = self.map[self.map.index(node + motion)]
            if (n.x, n.y) not in self.obstacles:
                neighbors.append(n)
        return neighbors

    def h(self, node: Node, goal: Node) -> float:
            """
            Calculate heuristic.

            Parameters:
                node (Node): current node
                goal (Node): goal node

            Returns:
                h (float): heuristic function value of node
            """
            if self.heuristic_type == "manhattan":
                return abs(goal.x - node.x) + abs(goal.y - node.y)
            elif self.heuristic_type == "euclidean":
                return math.hypot(goal.x - node.x, goal.y - node.y)

    def extractPath(self):
        """
        Extract the path based on greedy policy.

        Return
        ----------
        cost: float
            the cost of planning path
        path: list
            the planning path
        """
        node = self.goal
        path = [node.current]
        cost, count = 0, 0
        while node != self.start:
            neighbors = [
                node_n for node_n in self.getNeighbor(node) if not
                self.collision_checker(node.current, node_n.current, "onestep")
            ]
            next_node = min(neighbors, key=lambda n: n.g)
            path.append(next_node.current)
            cost += self.cost(node, next_node)
            node = next_node
            count += 1
            if count == 1000:
                return cost, []
        return cost, list(reversed(path))

