"""
@file: d_star.py
@breif: Dynamic A* motion planning
@author: Yang Haodong, Wu Maojia
@update: 2024.6.23
"""
from .graph_search import GraphSearcher
from python_motion_planning.utils import Env, Node, Grid


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
    def __init__(self, current: tuple, parent: tuple, t: str, h: float, k: float) -> None:
        self.current = current
        self.parent = parent
        self.t = t
        self.h = h
        self.k = k

    def __add__(self, node):
        return DNode((self.x + node.x, self.y + node.y), 
                     self.parent, self.t, self.h + node.h, self.k)

    def __str__(self) -> str:
        return "----------\ncurrent:{}\nparent:{}\nt:{}\nh:{}\nk:{}\n----------" \
            .format(self.current, self.parent, self.t, self.h, self.k)

class DStar(GraphSearcher):
    """
    Class for D* motion planning.

    Parameters:
        start (tuple): start point coordinate
        goal (tuple): goal point coordinate
        env (Grid): environment

    Examples:
        >>> import python_motion_planning as pmp
        >>> planner = pmp.DStar((5, 5), (45, 25), pmp.Grid(51, 31))
        >>> cost, path, _ = planner.plan()     # planning results only
        >>> planner.plot.animation(path, str(planner), cost)  # animation
        >>> planner.run()       # run both planning and animation

    References:
        [1]Optimal and Efficient Path Planning for Partially-Known Environments
    """
    def __init__(self, start: tuple, goal: tuple, env: Grid) -> None:
        super().__init__(start, goal, env, None)
        self.start = DNode(start, None, 'NEW', float('inf'), float("inf"))
        self.goal = DNode(goal, None, 'NEW', 0, float('inf'))
        # allowed motions
        self.motions = [DNode(motion.current, None, None, motion.g, 0) for motion in self.env.motions]
        # OPEN list and EXPAND list
        self.OPEN = []
        self.EXPAND = []
        # record history infomation of map grids
        self.map = {s: DNode(s, None, 'NEW', float("inf"), float("inf")) for s in self.env.grid_map}
        self.map[self.goal.current] = self.goal
        self.map[self.start.current] = self.start
        # intialize OPEN list
        self.insert(self.goal, 0)

    def __str__(self) -> str:
        return "Dynamic A*(D*)"

    def plan(self) -> tuple:
        """
        D* static motion planning function.

        Returns:
            cost (float): path cost
            path (list): planning path
            _ (None): None
        """
        while True:
            self.processState()
            if self.start.t == 'CLOSED':
                break
        cost, path = self.extractPath(self.map)
        return cost, path, None

    def run(self) -> None:
        """
        Running both plannig and animation.
        """
        # static planning
        cost, path, _ = self.plan()

        # animation
        self.plot.connect('button_press_event', self.OnPress)
        self.plot.animation(path, str(self), cost=cost)

    def OnPress(self, event) -> None:
        """
        Mouse button callback function.

        Parameters:
            event (MouseEvent): mouse event
        """
        x, y = int(event.xdata), int(event.ydata)
        if x < 0 or x > self.env.x_range - 1 or y < 0 or y > self.env.y_range - 1:
            print("Please choose right area!")
        else:
            if (x, y) not in self.obstacles:
                print("Add obstacle at: ({}, {})".format(x, y))
                # update obstacles
                self.obstacles.add((x, y))
                self.env.update(self.obstacles)

                # move from start to goal, replan locally when meeting collisions
                node = self.start
                self.EXPAND, path, cost = [], [], 0
                while node != self.goal:
                    node_parent = self.map[node.parent]
                    if self.isCollision(node, node_parent):
                        self.modify(node, node_parent)
                        continue
                    path.append(node.current)
                    cost += self.cost(node, node_parent)
                    node = node_parent

                self.plot.clean()
                self.plot.animation(path, str(self), cost, self.EXPAND)

            self.plot.update()

    def extractPath(self, closed_list: dict) -> tuple:
        """
        Extract the path based on the CLOSED list.

        Parameters:
            closed_list (dict): CLOSED list

        Returns:
            cost (float): the cost of planning path
            path (list): the planning path
        """
        cost = 0
        node = self.start
        path = [node.current]
        while node != self.goal:
            node_parent = closed_list[node.parent]
            cost += self.cost(node, node_parent)
            node = node_parent
            path.append(node.current)

        return cost, path

    def processState(self) -> float:
        """
        Broadcast dynamic obstacle information.

        Returns:
            min_k (float): minimum k value of map
        """
        # get node in OPEN list with min k value
        node = self.min_state
        self.EXPAND.append(node)

        if node is None:
            return -1

        # record the min k value of this iteration
        k_old = self.min_k
        # move node from OPEN list to CLOSED list
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
                        # Condition: LOWER happened in OPEN list (s), s should be explored again
                        self.insert(node, node.h)
                    else:
                        if node_n.parent != node.current and \
                            node.h > node_n.h + self.cost(node, node_n) and \
                            node_n.t == 'CLOSED' and \
                            node_n.h > k_old:
                            # Condition: LOWER happened in CLOSED list (s_n), s_n should be explored again
                            self.insert(node_n, node_n.h)
        return self.min_k

    @property
    def min_state(self) -> DNode:
        """
        Choose the node with the minimum k value in OPEN list.
        """
        if not self.OPEN:
            return None
        return min(self.OPEN, key=lambda node: node.k)

    @property
    def min_k(self) -> float:
        """
        Choose the minimum k value for nodes in OPEN list.
        """
        return self.min_state.k

    def insert(self, node: DNode, h_new: float) -> None:
        """
        Insert node into OPEN list.

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
        Delete node from OPEN list.

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
            n = self.map[(node + motion).current]
            if not self.isCollision(node, n):
                neighbors.append(n)
        return neighbors
