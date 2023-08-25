'''
@file: global_planner.py
@breif: global planner application entry
@author: Winter
@update: 2023.3.2
'''
from utils import Grid, Map, SearchFactory
from local_planner import DWA

if __name__ == '__main__':
    # build environment
    start = (2, 2, 0)
    goal = (11, 18, 0)
    env = Grid(51, 31)

    # creat planner
    planner = DWA(start, goal, env)
    planner.run()
   