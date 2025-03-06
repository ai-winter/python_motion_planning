"""
@file: control_factory.py
@breif: Facotry class for local planner.
@author: Winter
@update: 2023.10.24
"""
from python_motion_planning.local_planner import *

class ControlFactory(object):
    def __init__(self) -> None:
        pass

    def __call__(self, planner_name, **config):
        if planner_name == "dwa":
            return DWA(**config)
        elif planner_name == "pid":
            return PID(**config)
        elif planner_name == "apf":
            return APF(**config)
        elif planner_name == "rpp":
            return RPP(**config)
        elif planner_name == "lqr":
            return LQR(**config)
        elif planner_name == "mpc":
            return MPC(**config)
        else:
            raise ValueError("The `planner_name` must be set correctly.")