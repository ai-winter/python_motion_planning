"""
@file: aco.py
@breif: Ant Colony Optimization(ACO) motion planning
@author: Winter
@update: 2023.1.13
"""
import math
import random
from bisect import bisect_left

from python_motion_planning.common.structure import Env, Node
from python_motion_planning.planner import Planner

class ACO(Planner):
    """
    Class for Ant Colony Optimization(ACO) motion planning.

    Parameters:
        start (tuple): start point coordinate
        goal (tuple): goal point coordinate
        env (Env): environment
        heuristic_type (str): heuristic function type, default is euclidean
        n_ants (int): number of ants
        alpha (float): pheromone and heuristic factor weight coefficient
        beta (float): pheromone and heuristic factor weight coefficient
        rho (float): evaporation coefficient
        Q (float): pheromone gain
        max_iter (int): maximum iterations

    Examples:
        >>> from python_motion_planning.utils import Grid
        >>> from evolutionary_search import ACO
        >>> start = (5, 5)
        >>> goal = (45, 25)
        >>> env = Grid(51, 31)
        >>> planner = ACO(start, goal, env)
        >>> planner.run()

    References:
        [1] Ant Colony Optimization: A New Meta-Heuristic
    """
    def __init__(self, start: tuple, goal: tuple, env: Env, heuristic_type: str = "euclidean", 
        n_ants: int = 30, alpha: float = 1.0, beta: float = 5.0, rho: float = 0.1, Q: float = 1.0,
        max_iter: int = 50) -> None:
        super().__init__(start, goal, env)
        # allowed motions
        self.motions = [
            Node((-1, 0), None, 1, None), Node((-1, 1),  None, math.sqrt(2), None),
            Node((0, 1),  None, 1, None), Node((1, 1),   None, math.sqrt(2), None),
            Node((1, 0),  None, 1, None), Node((1, -1),  None, math.sqrt(2), None),
            Node((0, -1), None, 1, None), Node((-1, -1), None, math.sqrt(2), None)
        ]
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.max_iter = max_iter
        self.heuristic_type = heuristic_type

    def __str__(self) -> str:
        return "Ant Colony Optimization(ACO)"

    class Ant:
        def __init__(self) -> None:
            self.reset()
        
        def reset(self):
            self.found_goal = False
            self.current_node = None
            self.path = []
            self.steps = 0

    def plan(self):
        """
        Ant Colony Optimization(ACO) motion plan function.

        Returns:
            cost (float): path cost
            path (list): planning path
        """
        best_length_list, best_path = [], None

        # pheromone initialization
        pheromone_edges = {}
        for i in range(self.env.x_range):
            for j in  range(self.env.y_range):
                if (i, j) in self.obstacles:
                    continue
                cur_node = Node((i, j), (i, j), 0, 0)
                for node_n in self.getNeighbor(cur_node):
                    pheromone_edges[(cur_node, node_n)] = 1.0

        # heuristically set max steps
        max_steps = self.env.x_range * self.env.y_range / 2 + max(self.env.x_range, self.env.y_range)

        # main loop
        cost_list = []
        for _ in range(self.max_iter):
            ants_list = []
            for _ in range(self.n_ants):
                ant = self.Ant()
                ant.current_node = self.start
                while ant.current_node is not self.goal and ant.steps < max_steps:
                    ant.path.append(ant.current_node)

                    # candidate
                    prob_sum = 0.0
                    next_positions, next_probabilities = [], []
                    for node_n in self.getNeighbor(ant.current_node):                
                        # existed
                        if node_n in ant.path:
                            continue
                        
                        node_n.parent = ant.current_node.current

                        # goal found
                        if node_n == self.goal:
                            ant.path.append(node_n)
                            ant.found_goal = True
                            break

                        next_positions.append(node_n)
                        prob_new = pheromone_edges[(ant.current_node, node_n)] ** self.alpha \
                                    * (1.0 / self.h(node_n, self.goal)) ** self.beta
                        next_probabilities.append(prob_new)
                        prob_sum = prob_sum + prob_new
                    
                    if prob_sum == 0 or ant.found_goal:
                        break

                    # roulette selection
                    next_probabilities = list(map(lambda prob: prob / prob_sum, next_probabilities))
                    p0, cp = 0, []
                    for prob in next_probabilities:
                        p0 = p0 + prob
                        cp.append(p0)
                    ant.current_node = next_positions[bisect_left(cp, random.random())]

                    ant.steps = ant.steps + 1

                ants_list.append(ant)

            # pheromone deterioration
            for key, _ in pheromone_edges.items():
                pheromone_edges[key] *= (1 - self.rho)
            
            # pheromone update based on successful ants
            bpl, bp = float("inf"), None
            for ant in ants_list:
                if ant.found_goal:
                    if len(ant.path) < bpl:
                        bpl, bp = len(ant.path), ant.path
                    c = self.Q / len(ant.path)
                    for i in range(len(ant.path) - 1):
                        pheromone_edges[(ant.path[i], ant.path[i + 1])] += c
            
            if bpl < float("inf"):
                best_length_list.append(bpl)

            if len(best_length_list) > 0:
                cost_list.append(min(best_length_list))
                if bpl <= min(best_length_list):
                    best_path = bp

        if best_path:
            cost, path = self.extractPath(best_path)
            return cost, path, cost_list
        return [], [], []

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

    def getNeighbor(self, node: Node) -> list:
        """
        Find neighbors of node.

        Parameters:
            node (Node): current node

        Returns:
            neighbors (list): neighbors of current node
        """
        return [
            node + motion for motion in self.motions if not
            self.collision_checker(node.current, (node + motion).current, "onestep")
        ]

    def extractPath(self, closed_set):
        """
        Extract the path based on the CLOSED set.

        Parameters:
            closed_set (list): CLOSED set

        Returns:
            cost (float): path cost
            path (list): planning path
        """
        cost = 0
        node = closed_set[closed_set.index(self.goal)]
        path = [node.current]
        while node != self.start:
            node_parent = closed_set[closed_set.index(Node(node.parent, None, None, None))]
            cost += self.dist(node, node_parent)
            node = node_parent
            path.append(node.current)
        return cost, list(reversed(path))