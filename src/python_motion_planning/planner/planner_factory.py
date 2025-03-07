"""
@file: planner_factory.py
@breif: Factory class for path planner.
@author: Winter
@update: 2023.3.2
"""
from .graph_search import *
from .sample_search import *
from .evolutionary_search import *

class PlannerFactory(object):
    def __init__(self) -> None:
        pass

    def __call__(self, planner_name, **config):
        if planner_name == "a_star":
            return AStar(**config)
        elif planner_name == "dijkstra":
            return Dijkstra(**config)
        elif planner_name == "gbfs":
            return GBFS(**config)
        elif planner_name == "bi_a_star":
            return BiAStar(**config)
        elif planner_name == "jps":
            return JPS(**config)
        elif planner_name == "d_star":
            return DStar(**config)
        elif planner_name == "lpa_star":
            return LPAStar(**config)
        elif planner_name == "d_star_lite":
            return DStarLite(**config)
        elif planner_name == "voronoi":
            return VoronoiPlanner(**config)
        elif planner_name == "theta_star":
            return ThetaStar(**config)
        elif planner_name == "lazy_theta_star":
            return LazyThetaStar(**config)
        elif planner_name == "s_theta_star":
            return SThetaStar(**config)
        elif planner_name == "rrt":
            return RRT(**config)
        elif planner_name == "rrt_connect":
            return RRTConnect(**config)
        elif planner_name == "rrt_star":
            return RRTStar(**config)
        elif planner_name == "informed_rrt":
            return InformedRRT(**config)
        elif planner_name == "aco":
            return ACO(**config)
        elif planner_name == "pso":
            return PSO(**config)
        else:
            raise ValueError("The `planner_name` must be set correctly.")