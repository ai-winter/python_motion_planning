"""
@file: anya.py
@breif: Anya motion planning
@author: Wu Maojia
@update: 2024.3.17
"""
import heapq

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
            self.right = round(right)

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
    def __init__(self, interval: AnyaInterval, root: Point2D, parent = None, goal: Point2D = None) -> None:
        assert isinstance(parent, AnyaNode) or parent is None

        self.interval = interval
        self.root = root
        self.parent = parent
        self.start = self.parent.start if parent is not None else self.root
        self.goal = self.parent.goal if parent is not None else goal
        self.eps = self.interval.eps

        if parent is None:
            self.g = 0
        else:
            self.g = parent.g + root.dist(parent.root)

        self.h = self.heuristic
        self.f = self.g + self.h

    def __eq__(self, other) -> bool:
        if not isinstance(other, AnyaNode):
            return False
        return self.interval == other.interval and self.root == other.root

    def __lt__(self, other) -> bool:
        return self.f < other.f

    def __hash__(self) -> int:
        return hash((self.interval, self.root))

    def __str__(self) -> str:
        return "AnyaNode({!s}, {!s})".format(self.interval, self.root)

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def heuristic(self) -> float:
        # mirror the goal point if it is on the opposite side of the interval
        if ((self.root.y > self.interval.row and self.goal.y > self.interval.row) or
                (self.root.y < self.interval.row and self.goal.y < self.interval.row)):
            self.goal.y = 2 * self.interval.row - self.goal.y

        # the intersection point of the line connecting the root and the goal with the interval
        x = self.root.x + (self.goal.x - self.root.x) * (self.interval.row - self.root.y) / (self.goal.y - self.root.y)
        y = self.interval.row

        # restrict the intersection point to the interval
        if x < self.interval.left:
            x = self.interval.left
        elif x > self.interval.right:
            x = self.interval.right

        intersection = Point2D(x, y)

        return intersection.dist(self.root) + intersection.dist(self.goal)

    @property
    def x(self) -> float:
        return self.root.x

    @property
    def y(self) -> float:
        return self.root.y


class AnyaExpander(object):
    """
    Class for Anya expansion policy.

    Parameters:
    """
    def __init__(self, env) -> None:
        self.env = env

    def generate_successors(self, node: AnyaNode) -> list:
        if node.parent is None:     # start node has no parent
            return self.generate_start_successors(node)

        left = node.interval.left
        right = node.interval.right
        row = node.interval.row
        root = node.root
        left_endpoint = Point2D(left, row)
        right_endpoint = Point2D(right, row)
        successors = []

        if row == root.y:    # if node is flat
            # get the endpoint of the interval farthest from root
            if left_endpoint.dist(root) > right_endpoint.dist(root):
                endpoint = left_endpoint
            else:
                endpoint = right_endpoint

            # generate successors from the endpoint
            successors += self.generate_flat_successors(endpoint, node)

            # if the endpoint is a turning point on a taut local path beginning at root
            if self.is_corner_point(endpoint):
                successors += self.generate_cone_successors(left_endpoint, right_endpoint, node)

        else:   # if node is not flat, it must be a cone
            successors += self.generate_cone_successors(left_endpoint, right_endpoint, node)
            if self.is_corner_point(left_endpoint):
                successors += self.generate_flat_successors(left_endpoint, node)
                successors += self.generate_cone_successors(left_endpoint, left_endpoint, node)
            if self.is_corner_point(right_endpoint):
                successors += self.generate_flat_successors(right_endpoint, node)
                successors += self.generate_cone_successors(right_endpoint, right_endpoint, node)

        return successors

    def generate_start_successors(self, node: AnyaNode) -> list:
        root = node.root
        successors = []

        # scan left
        left = self.scan_row_left(root)
        if left is not None:
            intervals = self.split_interval(AnyaInterval(left, root.x, root.y, eps=self.env.eps))
            for interval in intervals:
                successors.append(AnyaNode(interval, root, node))

        # scan right
        right = self.scan_row_right(root)
        if right is not None:
            intervals = self.split_interval(AnyaInterval(root.x, right, root.y, eps=self.env.eps))
            for interval in intervals:
                successors.append(AnyaNode(interval, root, node))

        # scan the row above
        point_up = Point2D(root.x, root.y+1)
        left, right = self.scan_row_left(point_up), self.scan_row_right(point_up)
        if left is not None and right is not None:
            intervals = self.split_interval(AnyaInterval(left, right, point_up.y, eps=self.env.eps))
            for interval in intervals:
                successors.append(AnyaNode(interval, root, node))

        # scan the row below
        point_down = Point2D(root.x, root.y-1)
        left, right = self.scan_row_left(point_down), self.scan_row_right(point_down)
        if left is not None and right is not None:
            intervals = self.split_interval(AnyaInterval(left, right, point_down.y, eps=self.env.eps))
            for interval in intervals:
                successors.append(AnyaNode(interval, root, node))

        return successors

    def generate_flat_successors(self, point: Point2D, node: AnyaNode) -> list:
        root = node.root
        direction = 1 if point.x > root.x else -1
        current_point = point
        next_point = Point2D(current_point.x + direction, current_point.y)
        successors = []

        while not self.is_corner_point(current_point) and next_point.to_tuple not in self.env.obstacles:
            current_point = next_point
            next_point = Point2D(current_point.x + direction, current_point.y)

        if root.y == point.y:
            if current_point.x < root.x:
                successors.append(AnyaNode(
                        AnyaInterval(current_point.x, point.x, point.y, eps=self.env.eps),
                        root, node))
            else:
                successors.append(AnyaNode(
                        AnyaInterval(point.x, current_point.x, point.y, eps=self.env.eps),
                        root, node))
        else:
            if current_point.x < root.x:
                successors.append(AnyaNode(
                        AnyaInterval(current_point.x, point.x, point.y, eps=self.env.eps),
                        point, node))
            else:
                successors.append(AnyaNode(
                        AnyaInterval(point.x, current_point.x, point.y, eps=self.env.eps),
                        point, node))

        return successors

    def generate_cone_successors(self, left: Point2D, right: Point2D, node: AnyaNode) -> list:
        root = node.root
        successors = []

        if left.y == right.y and right.y == root.y:
            # get the endpoint of the interval farthest from root
            if left.dist(root) > right.dist(root):
                root_new = left
            else:
                root_new = right

            # scan the row above
            point_up = Point2D(root_new.x, root_new.y+1)
            left, right = self.scan_row_left(point_up), self.scan_row_right(point_up)
            if left is not None and right is not None:
                intervals = self.split_interval(AnyaInterval(left, right, point_up.y, eps=self.env.eps))
                for interval in intervals:
                    successors.append(AnyaNode(interval, root_new, node))

            # scan the row below
            point_down = Point2D(root_new.x, root_new.y-1)
            left, right = self.scan_row_left(point_down), self.scan_row_right(point_down)
            if left is not None and right is not None:
                intervals = self.split_interval(AnyaInterval(left, right, point_down.y, eps=self.env.eps))
                for interval in intervals:
                    successors.append(AnyaNode(interval, root_new, node))

        elif left == right:
            root_new = left     # left and right endpoints are the same

            # scan the row above
            point_up = Point2D(root_new.x, root_new.y+1)
            left, right = self.scan_row_left(point_up), self.scan_row_right(point_up)
            if left is not None and right is not None:
                intervals = self.split_interval(AnyaInterval(left, right, point_up.y, eps=self.env.eps))
                for interval in intervals:
                    successors.append(AnyaNode(interval, root_new, node))

            # scan the row below
            point_down = Point2D(root_new.x, root_new.y-1)
            left, right = self.scan_row_left(point_down), self.scan_row_right(point_down)
            if left is not None and right is not None:
                intervals = self.split_interval(AnyaInterval(left, right, point_down.y, eps=self.env.eps))
                for interval in intervals:
                    successors.append(AnyaNode(interval, root_new, node))

        else:
            # TODO: linear projection or directly up and down?
            root_new = root
            row_up = left.y + 1
            up_left = self.scan_row_left(Point2D(left.x, row_up))
            up_right = self.scan_row_right(Point2D(right.x, row_up))
            if up_left is not None and up_right is not None:
                # if the new interval contains any obstacle, discard it
                any_obstacle = False
                for x in range(up_left, up_right+1):
                    if (x, row_up) in self.env.obstacles:
                        any_obstacle = True
                        break
                if not any_obstacle:
                    intervals = self.split_interval(AnyaInterval(up_left, up_right, row_up, eps=self.env.eps))
                    for interval in intervals:
                        successors.append(AnyaNode(interval, root_new, node))

            row_down = right.y - 1
            down_left = self.scan_row_left(Point2D(left.x, row_down))
            down_right = self.scan_row_right(Point2D(right.x, row_down))
            if down_left is not None and down_right is not None:
                # if the new interval contains any obstacle, discard it
                any_obstacle = False
                for x in range(down_left, down_right+1):
                    if (x, row_down) in self.env.obstacles:
                        any_obstacle = True
                        break
                if not any_obstacle:
                    intervals = self.split_interval(AnyaInterval(down_left, down_right, row_down, eps=self.env.eps))
                    for interval in intervals:
                        successors.append(AnyaNode(interval, root_new, node))

        return successors

    def scan_row_left(self, point: Point2D) -> int:
        # TODO: half-closed interval
        if point.to_tuple in self.env.obstacles:
            return None
        left = point.x
        while (left - 1, point.y) not in self.env.obstacles:
            left -= 1
        return left

    def scan_row_right(self, point: Point2D) -> int:
        # TODO: half-closed interval
        if point.to_tuple in self.env.obstacles:
            return None
        right = point.x
        while (right + 1, point.y) not in self.env.obstacles:
            right += 1
        return right

    def split_interval(self, interval: AnyaInterval) -> list:
        traversable_intervals = []
        intervals = []

        # split by obstacles
        left = interval.left
        right = interval.left
        while right <= interval.right:
            if (right, interval.row) in self.env.obstacles:
                if right > left:
                    traversable_intervals.append(AnyaInterval(left, right-1, interval.row, eps=self.env.eps))
                left = right + 1
            right += 1
        if right > left:
            traversable_intervals.append(AnyaInterval(left, right-1, interval.row, eps=self.env.eps))

        # split by corner points
        for traversable_interval in traversable_intervals:
            left = traversable_interval.left
            for right in range(int(traversable_interval.left), int(traversable_interval.right+1)):
                if self.is_corner_point(Point2D(right, traversable_interval.row)):
                    intervals.append(AnyaInterval(left, right, traversable_interval.row, eps=self.env.eps))
                    left = right
            intervals.append(AnyaInterval(left, traversable_interval.right, traversable_interval.row, eps=self.env.eps))

        return intervals

    def is_corner_point(self, point: Point2D) -> bool:
        if point.to_tuple in self.env.obstacles:
            return False
        directions = [(1, 1), (-1, -1), (-1, 1), (1, -1)]
        for direction in directions:
            if ((point.x+direction[0], point.y+direction[1]) in self.env.obstacles
                    and (point.x, point.y+direction[1]) not in self.env.obstacles
                    and (point.x+direction[0], point.y) not in self.env.obstacles):
                return True
        return False


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
        self.start = start
        self.goal = goal
        self.expander = AnyaExpander(env)

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
        OPEN = []
        heapq.heappush(OPEN, AnyaNode(AnyaInterval(self.start[0], self.start[0], self.start[1], eps=self.env.eps),
                                      root=Point2D.from_tuple(self.start), parent=None, goal=Point2D.from_tuple(self.goal)))
        CLOSED = []
        while OPEN:
            node = heapq.heappop(OPEN)

            if node in CLOSED:
                continue

            if node.interval.contains(Point2D.from_tuple(self.goal)):
                CLOSED.append(node)
                cost, path = self.extractPath(CLOSED)
                return cost, path, CLOSED

            for successor in self.expander.generate_successors(node):
                if successor in CLOSED:
                    continue
                heapq.heappush(OPEN, successor)

            CLOSED.append(node)

        cost, path = self.extractPath(CLOSED)
        return cost, path, CLOSED

        # return [], [], []

    def extractPath(self, closed_set: list):
        path = [self.goal]
        node = closed_set[-1]
        cost = node.root.dist(Point2D.from_tuple(self.goal))
        while node.parent is not None:
            path.append(node.root.to_tuple)
            cost += node.root.dist(node.parent.root)
            node = node.parent

        return cost, path

    def run(self):
        """
        Running both planning and animation.
        """
        cost, path, expand = self.plan()
        self.plot.animation(path, str(self), cost, expand)