"""
@file: types.py
@breif: macro definition of types of plots in maps
@author: Yang Haodong, Wu Maojia
@update: 2025.3.28
"""
from enum import Enum

class TYPES(Enum):
    # environment macro
    FREE = 0
    OBSTACLE = 1
    START = 2
    GOAL = 3
    CUSTOM = 4
    INFLATION = 5