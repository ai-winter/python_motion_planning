"""
@file: rrt_star.py
@breif: RRT-Star motion planning
@author: Yang Haodong, Wu Maojia
@update: 2024.6.23
"""
from .rrt import RRT
from python_motion_planning.utils import Env, Node, Map


class RRTStar(RRT):
    """
    Class for RRT-Star motion planning.

    Parameters:
        start (tuple): start point coordinate
        goal (tuple): goal point coordinate
        env (Map): environment
        max_dist (float): Maximum expansion distance one step
        sample_num (int): Maximum number of sample points
        r (float): optimization radius
        goal_sample_rate (float): heuristic sample

    Examples:
        >>> import python_motion_planning as pmp
        >>> planner = pmp.RRTStar((5, 5), (45, 25), pmp.Map(51, 31))
        >>> cost, path, expand = planner.plan()     # planning results only
        >>> planner.plot.animation(path, str(planner), cost, expand)  # animation
        >>> planner.run()       # run both planning and animation

    References:
        [1] Sampling-based algorithms for optimal motion planning
    """
    def __init__(self, start: tuple, goal: tuple, env: Map, max_dist: float = 0.5,
                 sample_num: int = 10000, r: float = 10.0, goal_sample_rate: float = 0.05) -> None:
        super().__init__(start, goal, env, max_dist, sample_num, goal_sample_rate)
        # optimization radius
        self.r = r
    
    def __str__(self) -> str:
        return "RRT*"
    
    def getNearest(self, node_list: list, node: Node) -> Node:
        """
        Get the node from `node_list` that is nearest to `node` with optimization.

        Parameters:
            node_list (list): exploring list
            node (Node): currently generated node

        Returns:
            node (Node): nearest node
        """
        node_new = super().getNearest(node_list, node)
        if node_new:
            #  rewire optimization
            for node_n in node_list:
                #  inside the optimization circle
                new_dist = self.dist(node_n, node_new)
                if new_dist < self.r:
                    cost = node_n.g + new_dist
                    #  update new sample node's cost and parent
                    if node_new.g > cost and not self.isCollision(node_n, node_new):
                        node_new.parent = node_n.current
                        node_new.g = cost
                    else:
                        #  update nodes' cost inside the radius
                        cost = node_new.g + new_dist
                        if node_n.g > cost and not self.isCollision(node_n, node_new):
                            node_n.parent = node_new.current
                            node_n.g = cost
                else:
                    continue
            return node_new
        else:
            return None 
        