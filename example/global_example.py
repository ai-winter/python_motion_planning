'''
@file: global_planner.py
@breif: global planner application entry
@author: Winter
@update: 2023.3.2
'''
import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))
from utils import Grid, Map, SearchFactory


if __name__ == '__main__':
    '''
    path searcher constructor
    '''
    search_factory = SearchFactory()
    
    '''
    graph search
    '''
    # build environment
    start = (5, 5)
    goal = (45, 25)
    env = Grid(51, 31)

    # creat planner
    # planner = search_factory("a_star", start=start, goal=goal, env=env)
    # planner = search_factory("dijkstra", start=start, goal=goal, env=env)
    # planner = search_factory("gbfs", start=start, goal=goal, env=env)
    # planner = search_factory("jps", start=start, goal=goal, env=env)
    # planner = search_factory("d_star", start=start, goal=goal, env=env)
    # planner = search_factory("lpa_star", start=start, goal=goal, env=env)
    # planner = search_factory("d_star_lite", start=start, goal=goal, env=env)
    planner = search_factory("voronoi", start=start, goal=goal, env=env, n_knn=4,
                                max_edge_len=10.0, inflation_r=1.0)
    
    # animation
    planner.run()
    # from scipy.spatial import cKDTree
    # import numpy as np  
    # points_ref = np.array([(1, 1), (3, 3), (4, 4), (5, 4), (6, 6)])
    # tree = cKDTree(points_ref)
    # _, idx = tree.query((3, 3), k=4)
    # print(points_ref)

    # ========================================================

    '''
    sample search
    '''
    # # build environment
    # start = (18, 8)
    # goal = (37, 18)
    # env = Map(51, 31)

    # # creat planner
    # planner = search_factory("rrt", start=start, goal=goal, env=env, max_dist=0.5, sample_num=10000)
    # # planner = search_factory("rrt_connect", start=start, goal=goal, env=env, max_dist=0.5, sample_num=10000)
    # # planner = search_factory("rrt_star", start=start, goal=goal, env=env, max_dist=0.5, r=10, sample_num=10000)
    # # planner = search_factory("informed_rrt", start=start, goal=goal, env=env, max_dist=0.5, r=12, sample_num=1500)

    # # animation
    # planner.run()

    # ========================================================

    '''
    evolutionary search
    '''
    # planner = search_factory("aco", start=start, goal=goal, env=env)
    # planner.run()