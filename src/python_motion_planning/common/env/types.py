"""
@file: types.py
@author: Wu Maojia, Yang Haodong
@update: 2025.10.3
"""
# from enum import IntEnum

# class TYPES(IntEnum):
class TYPES:
    """
    Macro definition of types of plots in maps. They must be integers in sequence of (0, 1, 2, ...).
    """
    FREE = 0
    OBSTACLE = 1
    START = 2
    GOAL = 3
    INFLATION = 4
    EXPAND = 5
    CUSTOM = 6