'''
@file: global_planner.py
@breif: global planner application entry
@author: Winter
@update: 2023.3.2
'''
from utils import Grid, Map, SearchFactory
from local_planner import DWA

if __name__ == '__main__':
    a = {"a": 2, "b":3}
    for k, v in a.items():
        a[k] /= 2
    print(a)
    # # build environment
    # start = (2, 2, 0)
    # goal = (11, 18, 0)
    # env = Grid(51, 31)

    # # creat planner
    # planner = DWA(start, goal, env)
    # planner.run()
   