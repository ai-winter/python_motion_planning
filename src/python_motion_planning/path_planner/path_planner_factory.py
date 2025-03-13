"""
@file: path_planner_factory.py
@breif: Factory class for path planner.
@author: Winter
@update: 2023.3.2
"""
from .graph_search import *
from .sample_search import *

class PathPlannerFactory(object):
    def __init__(self) -> None:
        pass

    def __call__(self, planner_name, env, params):
        if planner_name == "astar":
            return AStarPlanner(env, params)
        elif planner_name == "dijkstra":
            return DijkstraPlanner(env, params)
        elif planner_name == "gbfs":
            return GBFSPlanner(env, params)
        elif planner_name == "jps":
            return JPSPlanner(env, params)
        elif planner_name == "dstar":
            return DStarPlanner(env, params)
        elif planner_name == "lpa_star":
            return LPAStarPlanner(env, params)
        elif planner_name == "dstar_lite":
            return DStarLitePlanner(env, params)
        elif planner_name == "voronoi":
            return VoronoiPlanner(env, params)
        elif planner_name == "theta_star":
            return ThetaStarPlanner(env, params)
        elif planner_name == "lazy_theta_star":
            return LazyThetaStarPlanner(env, params)
        elif planner_name == "s_theta_star":
            return SThetaStarPlanner(env, params)
        elif planner_name == "rrt":
            return RRTPlanner(env, params)
        elif planner_name == "rrt_connect":
            return RRTConnectPlanner(env, params)
        elif planner_name == "rrt_star":
            return RRTStarPlanner(env, params)
        elif planner_name == "informed_rrt":
            return InformedRRTPlanner(env, params)
        else:
            raise ValueError("The `planner_name` must be set correctly.")