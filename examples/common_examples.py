"""
@file: common_examples.py
@breif: Examples of Python Motion Planning library
@author: Wu Maojia
@update: 2024.11.22
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from python_motion_planning import *

if __name__ == '__main__':
    # -------------global planners-------------
    plt = AStar(start=(5, 5), goal=(45, 25), env=Grid(51, 31))
    # plt = DStar(start=(5, 5), goal=(45, 25), env=Grid(51, 31))
    # plt = DStarLite(start=(5, 5), goal=(45, 25), env=Grid(51, 31))
    # plt = Dijkstra(start=(5, 5), goal=(45, 25), env=Grid(51, 31))
    # plt = GBFS(start=(5, 5), goal=(45, 25), env=Grid(51, 31))
    # plt = JPS(start=(5, 5), goal=(45, 25), env=Grid(51, 31))
    # plt = ThetaStar(start=(5, 5), goal=(45, 25), env=Grid(51, 31))
    # plt = LazyThetaStar(start=(5, 5), goal=(45, 25), env=Grid(51, 31))
    # plt = SThetaStar(start=(5, 5), goal=(45, 25), env=Grid(51, 31))
    # plt = LPAStar(start=(5, 5), goal=(45, 25), env=Grid(51, 31))
    # plt = VoronoiPlanner(start=(5, 5), goal=(45, 25), env=Grid(51, 31))

    # plt = RRT(start=(18, 8), goal=(37, 18), env=Map(51, 31))
    # plt = RRTConnect(start=(18, 8), goal=(37, 18), env=Map(51, 31))
    # plt = RRTStar(start=(18, 8), goal=(37, 18), env=Map(51, 31))
    # plt = InformedRRT(start=(18, 8), goal=(37, 18), env=Map(51, 31))

    # plt = ACO(start=(5, 5), goal=(45, 25), env=Grid(51, 31))
    # plt = PSO(start=(5, 5), goal=(45, 25), env=Grid(51, 31))

    plt.run()

    # -------------local planners-------------
    # plt = PID(start=(5, 5, 0), goal=(45, 25, 0), env=Grid(51, 31))
    # plt = DWA(start=(5, 5, 0), goal=(45, 25, 0), env=Grid(51, 31))
    # plt = APF(start=(5, 5, 0), goal=(45, 25, 0), env=Grid(51, 31))
    # plt = LQR(start=(5, 5, 0), goal=(45, 25, 0), env=Grid(51, 31))
    # plt = RPP(start=(5, 5, 0), goal=(45, 25, 0), env=Grid(51, 31))
    # plt = MPC(start=(5, 5, 0), goal=(45, 25, 0), env=Grid(51, 31))
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