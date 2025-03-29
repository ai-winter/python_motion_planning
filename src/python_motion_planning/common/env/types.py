"""
@file: types.py
@breif: macro definition of types of plots in maps
@author: Yang Haodong, Wu Maojia
@update: 2025.3.29
"""
from enum import IntEnum

class TYPES(IntEnum):
    """
    Macro definition of types of plots in maps. They must be integers in sequence of (0, 1, 2, ...).
    """
    FREE = 0
    OBSTACLE = 1
    START = 2
    GOAL = 3
    CUSTOM = 4
    INFLATION = 5