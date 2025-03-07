"""
@file: d_star.py
@breif: Dynamic A* motion planning
@author: Yang Haodong
@update: 2024.2.11
"""
import math

from python_motion_planning.planner import Planner
from python_motion_planning.common.utils import LOG
from python_motion_planning.common.structure import Node
from python_motion_planning.common.utils import Visualizer
from python_motion_planning.common.geometry import Point3d

class DNode(Node):
    """
    Class for D* nodes.

    Parameters:
        current (tuple): current coordinate
        parent (tuple): coordinate of parent node
        t (str): state of node, including `NEW` `OPEN` and `CLOSED`
        h (float): cost from goal to current node
        k (float): minimum cost from goal to current node in history
    """
    def __init__(self, current: Point3d, parent: Point3d, t: str, h: float, k: float) -> None:
        self.current = current
        self.parent = parent
        self.t = t
        self.h = h
        self.k = k

    def __add__(self, node):
        return DNode(
            Point3d(self.current.x() + node.current.x(),
                    self.current.y() + node.current.y(),
                    self.current.theta() + node.current.theta()
            ), self.parent, self.t, self.h + node.h, self.k
        )

    def __eq__(self, node: 'Node') -> bool:
        return (self.x == node.x and self.y == node.y)

    def __str__(self) -> str:
        return f"----------\ncurrent:({self.x}, {self.y})\nparent:({self.px}, {self.py})\nt:{self.t}\nh:{self.h}\nk:{self.k}----------"

class DStar(Planner):
    """
    Class for D* motion planning.

    Parameters:
        params (dict): parameters

    References:
        [1]Optimal and Efficient Path Planning for Partially-Known Environments
    """
    def __init__(self, params: dict) -> None:
        super().__init__(params)
        # allowed motions
        self.motions = [
            DNode(Point3d(-1,  0, 0), None, None, 1, 0), DNode(Point3d(-1,  1, 0), None, None, math.sqrt(2), 0),
            DNode(Point3d( 0,  1, 0), None, None, 1, 0), DNode(Point3d( 1,  1, 0), None, None, math.sqrt(2), 0),
            DNode(Point3d( 1,  0, 0), None, None, 1, 0), DNode(Point3d( 1, -1, 0), None, None, math.sqrt(2), 0),
            DNode(Point3d( 0, -1, 0), None, None, 1, 0), DNode(Point3d(-1, -1, 0), None, None, math.sqrt(2), 0)
        ]

        # OPEN set and EXPAND set
        self.OPEN = []
        self.EXPAND = []

        # record history infomation of map grids
        grid_coor = {Point3d(i, j, 0) for i in range(self.env.x_range) for j in range(self.env.y_range)}
        self.map = [DNode(s, None, 'NEW', float("inf"), float("inf")) for s in grid_coor]
        
        # updated path
        self.path = []
        
        # updated cost
        self.p_cost = None
        
        # visualizer
        self.visualizer = None

        # parameters
        for k, v in self.params["strategy"]["planner"].items():
            setattr(self, k, v)

    def __str__(self) -> str:
        return "Dynamic A*(D*)"

    def plan(self, start: Point3d, goal: Point3d):
        """
        D* static motion planning function.

        Returns:
            cost (float): path cost
            path (list): planning path
            expand (list): all nodes that planner has searched
        """
        self.start = DNode(start, None, 'NEW', float('inf'), float("inf"))
        self.goal = DNode(goal, None, 'NEW', 0, float('inf'))
        self.map[self.map.index(self.goal)] = self.goal
        self.map[self.map.index(self.start)] = self.start
        # intialize OPEN set
        self.insert(self.goal, 0)

        while True:
            self.processState()
            if self.start.t == 'CLOSED':
                break

        self.p_cost, self.path = self.extractPath(self.map)
        
        LOG.INFO(f"{str(self)} Planner Planning Successfully. Cost: {self.p_cost}")
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
            if (x, y) not in self.obstacles:
                LOG.INFO(f"Change position: x = {x}, y = {y}]")
                # update obstacles
                self.obstacles.append((x, y))
                self.env.update(self.obstacles)
                self.collision_checker.update(self.obstacles)

                # move from start to goal, replan locally when meeting collisions
                node = self.start
                self.EXPAND, self.path, self.p_cost = [], [], 0
                while node != self.goal:
                    node_parent = self.map[self.map.index(DNode(node.parent, None, None, None, None))]
                    if self.collision_checker(node.current, node_parent.current, "onestep"):
                        self.modify(node, node_parent)
                        continue
                    self.path.append(node.current)
                    self.p_cost += self.cost(node, node_parent)
                    node = node_parent

                LOG.INFO(f"{str(self)} Planner Planning Successfully. Cost: {self.p_cost}")

                # animation
                self.visualizer.clean()
                self.visualizer.plotGridMap(self.env)
                self.visualizer.plotGrids([
                    {"x": self.start.x, "y": self.start.y, "name": "start"},
                    {"x": self.goal.x, "y": self.goal.y, "name": "goal"},
                ] + [
                    {"x": n.x, "y": n.y, "name": "custom"}
                        for n in self.EXPAND if n != self.start and n != self.goal
                ])
                self.visualizer.setTitle(f"{str(self)}\ncost: {self.p_cost}")
                self.visualizer.plotPath(self.path)

            self.visualizer.update()

    def extractPath(self, closed_set):
        """
        Extract the path based on the CLOSED set.

        Parameters:
            closed_set (list): CLOSED set

        Returns:
            cost (float): the cost of planning path
            path (list): the planning path
        """
        cost = 0
        node = self.start
        path = [node.current]
        while node != self.goal:
            node_parent = closed_set[closed_set.index(DNode(node.parent, None, None, None, None))]
            cost += self.cost(node, node_parent)
            node = node_parent
            path.append(node.current)
        return cost, list(reversed(path))

    def processState(self) -> float:
        """
        Broadcast dynamic obstacle information.

        Returns:
            min_k (float): minimum k value of map
        """
        # get node in OPEN set with min k value
        node = self.min_state
        self.EXPAND.append(node)

        if node is None:
            return -1

        # record the min k value of this iteration
        k_old = self.min_k
        # move node from OPEN set to CLOSED set
        self.delete(node)  

        # k_min < h[x] --> x: RAISE state (try to reduce k value by neighbor)
        if k_old < node.h:
            for node_n in self.getNeighbor(node):
                if node_n.h <= k_old and node.h > node_n.h + self.cost(node, node_n):
                    # update h_value and choose parent
                    node.parent = node_n.current
                    node.h = node_n.h + self.cost(node, node_n)

        # k_min >= h[x] -- > x: LOWER state (cost reductions)
        if k_old == node.h:
            for node_n in self.getNeighbor(node):
                if node_n.t == 'NEW' or \
                    (node_n.parent == node.current and node_n.h != node.h + self.cost(node, node_n)) or \
                    (node_n.parent != node.current and node_n.h > node.h + self.cost(node, node_n)):
                    # Condition:
                    # 1) t[node_n] == 'NEW': not visited
                    # 2) node_n's parent: cost reduction
                    # 3) node_n find a better parent
                    node_n.parent = node.current
                    self.insert(node_n, node.h + self.cost(node, node_n))
        else:
            for node_n in self.getNeighbor(node):
                if node_n.t == 'NEW' or \
                    (node_n.parent == node.current and node_n.h != node.h + self.cost(node, node_n)):
                    # Condition:
                    # 1) t[node_n] == 'NEW': not visited
                    # 2) node_n's parent: cost reduction
                    node_n.parent = node.current
                    self.insert(node_n, node.h + self.cost(node, node_n))
                else:
                    if node_n.parent != node.current and \
                        node_n.h > node.h + self.cost(node, node_n):
                        # Condition: LOWER happened in OPEN set (s), s should be explored again
                        self.insert(node, node.h)
                    else:
                        if node_n.parent != node.current and \
                            node.h > node_n.h + self.cost(node, node_n) and \
                            node_n.t == 'CLOSED' and \
                            node_n.h > k_old:
                            # Condition: LOWER happened in CLOSED set (s_n), s_n should be explored again
                            self.insert(node_n, node_n.h)
        return self.min_k

    @property
    def min_state(self) -> DNode:
        """
        Choose the node with the minimum k value in OPEN set.
        """
        if not self.OPEN:
            return None
        return min(self.OPEN, key=lambda node: node.k)

    @property
    def min_k(self) -> float:
        """
        Choose the minimum k value for nodes in OPEN set.
        """
        return self.min_state.k

    def insert(self, node: DNode, h_new: float) -> None:
        """
        Insert node into OPEN set.

        Parameters:
            node (DNode): the node to insert
            h_new (float): new or better cost to come value
        """
        if node.t == 'NEW':         node.k = h_new
        elif node.t == 'OPEN':      node.k = min(node.k, h_new)
        elif node.t == 'CLOSED':    node.k = min(node.h, h_new)
        node.h, node.t = h_new, 'OPEN'
        self.OPEN.append(node)

    def delete(self, node: DNode) -> None:
        """
        Delete node from OPEN set.

        Parameters:
            node (DNode): the node to delete
        """
        if node.t == 'OPEN':
            node.t = 'CLOSED'
        self.OPEN.remove(node)

    def modify(self, node: DNode, node_parent: DNode) -> None:
        """
        Start processing from node.

        Parameters:
            node (DNode): the node to modify
            node_parent (DNode): the parent node of `node`
        """
        if node.t == 'CLOSED':
            self.insert(node, node_parent.h + self.cost(node, node_parent))
        while True:
            k_min = self.processState()
            if k_min >= node.h:
                break

    def getNeighbor(self, node: DNode) -> list:
        """
        Find neighbors of node.

        Parameters:
            node (DNode): current node

        Returns:
            neighbors (list): neighbors of current node
        """
        neighbors = []
        for motion in self.motions:
            n = self.map[self.map.index(node + motion)]
            if not self.collision_checker(node.current, n.current, "onestep"):
                neighbors.append(n)
        return neighbors
