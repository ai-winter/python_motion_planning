"""
@file: command.py
@breif: robot control command
@author: Winter
@update: 2025.1.8
"""
from collections import namedtuple

DiffCmd = namedtuple("DiffCmd", "v w")