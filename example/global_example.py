'''
@file: global_example.py
@breif: global planner application examples
@author: Winter, Wu Maojia
@update: 2024.2.2
'''
import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))
from utils import Grid, Map, SearchFactory
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Global Planner Example')
    parser.add_argument('--algorithm', '-a', type=str, default='a_star', help='global algorithm type')
    parser.add_argument('--start', '-s', type=tuple, default=(5, 5), help='start point coordinate')
    parser.add_argument('--goal', '-g', type=tuple, default=(45, 25), help='goal point coordinate')
    parser.add_argument('--env', '-e', type=tuple, default=(51,31), help='environment size')
    parser.add_argument('--n-knn', type=int, default=4, help='number of edges from one sampled point, for "voronoi"')
    parser.add_argument('--max-edge-len', type=float, default=10.0, help='maximum edge length, for "voronoi"')
    parser.add_argument('--inflation-r', type=float, default=1.0, help='inflation radius, for "voronoi"')
    parser.add_argument('--max-dist', type=float, default=0.5, help='maximum distance of sample points, '
                                                                    'for sample algorithms')
    parser.add_argument('--sample-num', type=int, default=10000, help='sample number, for sample algorithms')
    parser.add_argument('--optim-r', type=float, default=12, help='optimization radius, for "rrt_star"')

    parsed_args = parser.parse_args()
    assert parsed_args.algorithm in ["a_star", "dijkstra", "gbfs", "theta_star", "lazy_theta_star", "jps", "d_star",
                                     "lpa_star", "d_star_lite", "voronoi", "rrt", "rrt_connect", "rrt_star",
                                     "informed_rrt", "aco"], "Invalid algorithm type"

    return parsed_args


if __name__ == '__main__':
    args = get_args()

    '''
    path searcher constructor
    '''
    search_factory = SearchFactory()
    
    '''
    graph search
    '''
    # build environment
    start = args.start
    goal = args.goal
    if args.algorithm in ["a_star", "dijkstra", "gbfs", "theta_star", "lazy_theta_star", "jps", "d_star", "lpa_star",
                          "d_star_lite", "voronoi"]:    # graph search
        env = Grid(args.env[0], args.env[1])
    else:   # sample search and evolutionary search
        env = Map(args.env[0], args.env[1])

    # creat planner
    if args.algorithm == "voronoi":
        planner = search_factory(args.algorithm, start=start, goal=goal, env=env, n_knn=4, max_edge_len=10.0,
                                 inflation_r=1.0)
    elif args.algorithm in ["rrt", "rrt_connect"]:
        planner = search_factory(args.algorithm, start=start, goal=goal, env=env, max_dist=0.5, sample_num=10000)
    elif args.algorithm in ["rrt_star", "informed_rrt"]:
        planner = search_factory(args.algorithm, start=start, goal=goal, env=env, max_dist=0.5, r=12, sample_num=10000)
    else:   # ["a_star", "dijkstra", "gbfs", "theta_star", "lazy_theta_star", "jps", "d_star", "lpa_star",
        # "d_star_lite", "aco"]
        planner = search_factory(args.algorithm, start=start, goal=goal, env=env)
    
    # animation
    planner.run()