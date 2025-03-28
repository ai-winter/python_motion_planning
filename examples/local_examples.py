"""
@file: local_examples.py
@breif: local planner application examples
@author: Yang Haodong, Wu Maojia
@update: 2024.11.22
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from python_motion_planning.utils import Grid, ControlFactory

if __name__ == '__main__':
    '''
    local planner constructor
    '''
    control_factory = ControlFactory()
    
    # build environment
    start = (5, 5, 0)
    goal = (45, 25, 0)
    env = Grid(51, 31)

    # creat planner
    planner = control_factory("pid", start=start, goal=goal, env=env)

    # animation
    planner.run()