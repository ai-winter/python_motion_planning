'''
@file: local_planner.py
@breif: Base class for local planner.
@author: Winter
@update: 2023.3.2
'''
import math
import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))

from utils import Env, Planner, SearchFactory, Plot


class LocalPlanner(Planner):
    '''
    Base class for local planner.

    Parameters
    ----------
    start: tuple
        start point coordinate
    goal: tuple
        goal point coordinate
    env: Env
        environment
    heuristic_type: str
        heuristic function type, default is euclidean
    '''
    def __init__(self, start: tuple, goal: tuple, env: Env, heuristic_type: str="euclidean") -> None:
        # start and goal pose
        assert len(start) == 3 and len(goal) == 3, \
            "Start and goal parameters must be (x, y, theta)"
        self.start, self.goal = start, goal
        # heuristic type
        self.heuristic_type = heuristic_type
        # environment
        self.env = env
        # obstacles
        self.obstacles = self.env.obstacles
        # graph handler
        self.plot = Plot(start, goal, env)

        # global planner
        self.g_planner_ = None
        # search factory
        self.search_factory_ = SearchFactory()
    
    def dist(self, start: tuple, end: tuple) -> float:
        return math.hypot(end[0] - start[0], end[1] - start[1])
    
    def angle(self, start: tuple, end: tuple) -> float:
        return math.atan2(end[1] - start[1], end[0] - start[0])

    @property
    def g_planner(self):
        return str(self.g_planner_)
    
    @g_planner.setter
    def g_planner(self, **config):
        if hasattr(config, "planner_name"):
            self.g_planner_ = self.search_factory_(**config)
        else:
            raise RuntimeError("Please set planner name!")
    
    @property
    def g_path(self):
        '''
        [property]Global path.
        '''        
        if self.g_planner_ is None:
            raise AttributeError("Global path searcher is None, please set it first!")
        
        (cost, path), _ = self.g_planner_.plan()
        return path