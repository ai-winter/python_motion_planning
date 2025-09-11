"""
@file: common_examples.py
@breif: Examples of Python Motion Planning library
@author: Wu Maojia
@update: 2025.4.11
"""
import pandas as pd
import sys, os
sys.path.insert(0, '/Users/hermanbr/Master/IN5060/python_motion_planning/src')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from python_motion_planning import *

def plt_in_array(plt):
    cost, path, expand = plt.run()
    algorithm_name.append(plt.__str__())
    x_size.append(x)
    y_size.append(y)
    z_size.append(z)
    cost_array.append(cost)
    search_area_array.append(len(expand))

if __name__ == '__main__':
    # Create environment with custom obstacles
    x = 10
    y = 10
    z = 10
    grid_env = Grid(x, y, z)

    obstacles = grid_env.obstacles
    for i in range(x):
        for j in range(y):
            obstacles.add((i, j, 7))
            obstacles.add((i, j, 5))
            obstacles.add((i, j, 3))

    obstacles.remove((8,8,3))
    obstacles.remove((1,1,5))
    obstacles.remove((5,5,7))

    grid_env.update(obstacles=obstacles)
    #print(grid_env.obstacles)

    """
    map_env = Map(51, 31, 1)
    obs_rect = [
        [14, 12, 8, 2],
        [18, 22, 8, 3],
        [26, 7, 2, 12],
        [32, 14, 10, 2]
    ]
    obs_circ = [
        [7, 12, 3],
        [46, 20, 2],
        [15, 5, 2],
        [37, 7, 3],
        [37, 23, 3]
    ]
    map_env.update(obs_rect=obs_rect, obs_circ=obs_circ)

    """
    # -------------global planners-------------
    algorithm_name = []
    x_size = []
    y_size = []
    z_size = []
    cost_array = []
    search_area_array = []


    start_pos = (5, 5, 1)
    goal_pos = (5, 5,  8)
    



    plt = AStar(start=(5, 5, 1), goal=(5, 5, 8), env=grid_env)
    plt_in_array(plt)
    # plt = DStar(start=(5, 5), goal=(45, 25), env=grid_env)
    # plt = DStarLite(start=(5, 5), goal=(45, 25), env=grid_env)
    plt = Dijkstra(start=(1, 1, 1), goal=(7, 7, 7), env=grid_env)
    plt_in_array(plt)
    # plt = GBFS(start=(5, 5), goal=(45, 25), env=grid_env)
    # plt = JPS(start=(5, 5, 5), goal=(45, 25, 5), env=grid_env)
    # plt_in_array(plt)
    # plt = ThetaStar(start=(5, 5), goal=(45, 25), env=grid_env)
    # plt = LazyThetaStar(start=(5, 5), goal=(45, 25), env=grid_env)
    # plt = SThetaStar(start=(5, 5), goal=(45, 25), env=grid_env)
    # plt = LPAStar(start=(5, 5), goal=(45, 25), env=grid_env)
    # plt = VoronoiPlanner(start=(5, 5), goal=(45, 25), env=grid_env)

    # plt = RRT(start=(18, 8), goal=(37, 18), env=map_env)
    # plt = RRTConnect(start=(18, 8), goal=(37, 18), env=map_env)
    # plt = RRTStar(start=(18, 8), goal=(37, 18), env=map_env)
    # plt = InformedRRT(start=(18, 8), goal=(37, 18), env=map_env)

    # plt = ACO(start=(5, 5), goal=(45, 25), env=grid_env)
    # plt = PSO(start=(5, 5), goal=(45, 25), env=grid_env)

    # cost, path, expand = plt.run()
    # algorithm_name.append(plt.__str__())
    # x_size.append(x)
    # y_size.append(y)
    # z_size.append(z)
    # cost_array.append(cost)
    # search_area_array.append(len(expand))

    #Write to CSV file with panda 
    dict = {'algorithm': algorithm_name, 'x_size': x_size, 'y_size': y_size, 'z_size': z_size, 'cost': cost_array, 'search_area': search_area_array}
    data_frame = pd.DataFrame(dict)

    data_frame.to_csv('file1.csv', index=False)

 
    # -------------local planners-------------
    # plt = PID(start=(5, 5, 0), goal=(45, 25, 0), env=grid_env)
    # plt = DWA(start=(5, 5, 0), goal=(45, 25, 0), env=grid_env)
    # plt = APF(start=(5, 5, 0), goal=(45, 25, 0), env=grid_env)
    # plt = LQR(start=(5, 5, 0), goal=(45, 25, 0), env=grid_env)
    # plt = RPP(start=(5, 5, 0), goal=(45, 25, 0), env=grid_env)
    # plt = MPC(start=(5, 5, 0), goal=(45, 25, 0), env=grid_env)
    # plt.run()

    # -------------curve generators-------------
    # points = [(0, 0, 0), (10, 10, -90), (20, 5, 60), (30, 10, 120),
    #           (35, -5, 30), (25, -10, -120), (15, -15, 100), (0, -10, -90)]

    # plt = Dubins(step=0.1, max_curv=0.25)
    # plt = Bezier(step=0.1, offset=3.0)
    # plt = Polynomial(step=2, max_acc=1.0, max_jerk=0.5)
    # plt = ReedsShepp(step=0.1, max_curv=0.25)
    # plt = CubicSpline(step=0.1)
    # plt = BSpline(step=0.01, k=3)

    # plt.run(points)
