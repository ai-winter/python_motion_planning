'''
@file: lqr.py
@breif: Linear Quadratic Regulator(LQR) motion planning
@author: Winter
@update: 2024.1.12
'''
import os, sys
import numpy as np
from itertools import product
from scipy.spatial.distance import cdist

sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))

from .local_planner import LocalPlanner
from utils import Env, Robot

class LQR(LocalPlanner):
    '''
    Class for Linear Quadratic Regulator(LQR) motion planning.

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

    Examples
    ----------
    '''
    def __init__(self, start: tuple, goal: tuple, env: Env, heuristic_type: str = "euclidean") -> None:
        super().__init__(start, goal, env, heuristic_type)

        