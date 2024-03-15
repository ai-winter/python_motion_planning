"""
@file: anya.py
@breif: Anya motion planning
@author: Wu Maojia
@update: 2024.3.15
"""
from .graph_search import GraphSearcher
from python_motion_planning.utils import Env, Point2D


class AnyaInterval(object):
    """
    Class for Anya interval.

    Parameters:
        left (float): left endpoint
        right (float): right endpoint
        row (int): row index of the interval
        eps (float): tolerance for float comparison
    """
    def __init__(self, left: float, right: float, row: int, eps: float = 1e-6) -> None:
        self.left = left
        self.right = right
        self.row = row
        self.eps = eps

        self.left_is_discrete = self.__is_discrete(self.left)
        self.right_is_discrete = self.__is_discrete(self.right)
        if self.left_is_discrete:
            self.left = round(left)
        if self.right_is_discrete:
            self.left = round(right)

    def __eq__(self, other) -> bool:
        if not isinstance(other, AnyaInterval):
            return False
        return (abs(self.left - other.left) < self.eps and
                abs(self.right - other.right) < self.eps and
                self.row == other.row)

    def __hash__(self) -> int:
        return hash((self.left, self.right, self.row))

    def __len__(self) -> float:
        return self.right - self.left

    def __str__(self) -> str:
        return "AnyaInterval(left={:.2f}, right={:.2f}, row={:d})".format(self.left, self.right, self.row)

    def __repr__(self) -> str:
        return self.__str__()

    def __is_discrete(self, x: float) -> bool:
        return abs(x - round(x)) < self.eps

    def covers(self, other) -> bool:
        if self == other:
            return True
        return self.left <= other.left and self.right >= other.right and self.row == other.row

    def contains(self, point: Point2D) -> bool:
        return round(point.x) == self.row and self.left - self.eps <= point.y <= self.right + self.eps


class AnyaNode(object):
    """
    Class for Anya nodes.

    Parameters:
        interval (AnyaInterval): interval of the node
        root (Point2D): root point of the node
        parent (AnyaNode): parent node of the node
        eps (float): tolerance for float comparison
    """
    def __init__(self, interval: AnyaInterval, root: Point2D, parent=None, eps: float = 1e-6) -> None:
        assert isinstance(parent, AnyaNode) or parent is None

        self.interval = interval
        self.root = root
        self.parent = parent
        self.eps = eps

        if parent is None:
            self.g = 0
        else:
            self.g = parent.g + root.dist(parent.root)

    def __eq__(self, other) -> bool:
        if not isinstance(other, AnyaNode):
            return False
        return self.interval == other.interval and self.root == other.root

    def __hash__(self) -> int:
        return hash((self.interval, self.root))

    def __str__(self) -> str:
        return "AnyaNode({!s}, {!s})".format(self.interval, self.root)

    def __repr__(self) -> str:
        return self.__str__()

    def heuristic(self, goal: Point2D) -> float:
        # mirror the goal point if it is on the opposite side of the interval
        if ((self.root.y > self.interval.row and goal.y > self.interval.row) or
                (self.root.y < self.interval.row and goal.y < self.interval.row)):
            goal.y = 2 * self.interval.row - goal.y

        # the intersection point of the line connecting the root and the goal with the interval
        x = self.root.x + (goal.x - self.root.x) * (self.interval.row - self.root.y) / (goal.y - self.root.y)
        y = self.interval.row

        # restrict the intersection point to the interval
        if x < self.interval.left:
            x = self.interval.left
        elif x > self.interval.right:
            x = self.interval.right

        intersection = Point2D(x, y)

        return intersection.dist(self.root) + intersection.dist(goal)


class AnyaExpander(object):
    """
    Class for Anya expansion policy.

    Parameters:
    """
    def __init__(self) -> None:
        pass

    def generate_successors(self, node: AnyaNode) -> list:
        pass

    def generate_start_successors(self, point: Point2D) -> list:
        pass

    def generate_flat_successors(self, node: AnyaNode) -> list:
        pass

    def generate_cone_successors(self, node: AnyaNode) -> list:
        root_new = None

        if node.interval.row == node.root.y:
            # get the farthest point from root
            if node.interval.left.dist(node.root) < node.interval.right.dist(node.root):
                root_new = Point2D(node.interval.right, node.interval.row)
            else:
                root_new = Point2D(node.interval.left, node.interval.row)

            # points from an adjacent row
            point_up = Point2D(node.interval.left, node.interval.row + 1)
            point_down = Point2D(node.interval.right, node.interval.row - 1)

            # a maximum open interval, beginning at p and entirely observable from root_new


        elif len(node.interval) < node.eps:
            root_new = Point2D(node.interval.left, node.interval.row)

            # a point from an adjacent row


            # a maximum closed interval, beginning at p and entirely observable from root_new

        else:
            pass



class Anya(GraphSearcher):
    """
    Class for Anya motion planning.

    Parameters:
        start (tuple): start point coordinate
        goal (tuple): goal point coordinate
        env (Env): environment

    Examples:
        >>> from python_motion_planning.utils import Grid
        >>> from graph_search import Anya
        >>> start = (5, 5)
        >>> goal = (45, 25)
        >>> env = Grid(51, 31)
        >>> planner = Anya(start, goal, env)
        >>> planner.run()

    References:
        [1] An Optimal Any-Angle Pathfinding Algorithm
        [2] Optimal Any-Angle Pathfinding In Practice
    """
    def __init__(self, start: tuple, goal: tuple, env: Env) -> None:
        super().__init__(start, goal, env, None)
        self.start = DNode(start, None, 'NEW', float('inf'), float("inf"))
        self.goal = DNode(goal, None, 'NEW', 0, float('inf'))
        # record history infomation of map grids
        self.map = None
        # allowed motions
        self.motions = [DNode(motion.current, None, None, motion.g, 0) for motion in self.env.motions]
        # OPEN set and EXPAND set
        self.OPEN = []
        self.EXPAND = []

    def __str__(self) -> str:
        return "Anya"

    def plan(self):
        """
        Anya static motion planning function.

        Returns:
            cost (float): path cost
            path (list): planning path
            expand (list): all nodes that planner has searched
        """
        while True:
            self.processState()
            if self.start.t == 'CLOSED':
                break
        cost, path = self.extractPath(self.map)
        return cost, path, None

    def run(self):
        """
        Running both plannig and animation.
        """
        # intialize global information
        self.map = [DNode(s, None, 'NEW', float("inf"), float("inf")) for s in self.env.grid_map]
        self.map[self.map.index(self.goal)] = self.goal
        self.map[self.map.index(self.start)] = self.start

        # intialize OPEN set
        self.insert(self.goal, 0)

        # static planning
        cost, path, _ = self.plan()

        # animation
        self.plot.connect('button_press_event', self.OnPress)
        self.plot.animation(path, str(self), cost=cost)

    def OnPress(self, event):
        """
        Mouse button callback function.
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
                    node_parent = self.map[self.map.index(DNode(node.parent, None, None, None, None))]
                    if self.isCollision(node, node_parent):
                        self.modify(node, node_parent)
                        continue
                    path.append(node.current)
                    cost += self.cost(node, node_parent)
                    node = node_parent

                self.plot.clean()
                self.plot.animation(path, str(self), cost, self.EXPAND)

            self.plot.update()

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

        return cost, path

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
            if not self.isCollision(node, n):
                neighbors.append(n)
        return neighbors
