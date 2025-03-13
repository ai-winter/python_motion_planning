"""
@file: rrt_star_planner.py
@breif: RRT* path planning
@author: Winter
@update: 2023.1.18
"""
from .rrt_planner import RRTPlanner

from python_motion_planning.common.structure import Node, Env

class RRTStarPlanner(RRTPlanner):
    """
    Class for RRT* path planning.

    Parameters:
        env (Env): environment object
        params (dict): parameters

    References:
        [1] Sampling-based algorithms for optimal motion planning
    """
    def __init__(self, env: Env, params: dict) -> None:
        super().__init__(env, params)
    
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
                    if node_new.g > cost and not self.collision_checker(node_n.current, node_new.current):
                        node_new.parent = node_n.current
                        node_new.g = cost
                    else:
                        #  update nodes' cost inside the radius
                        cost = node_new.g + new_dist
                        if node_n.g > cost and not self.collision_checker(node_n.current, node_new.current):
                            node_n.parent = node_new.current
                            node_n.g = cost
                else:
                    continue
            return node_new
        else:
            return None 
        