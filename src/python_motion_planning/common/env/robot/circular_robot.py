"""
@file: circular_robot.py
@author: Wu Maojia
@update: 2025.10.3
"""
from typing import List, Dict, Tuple, Optional
import numpy as np

from python_motion_planning.common.utils.geometry import Geometry
from .base_robot import BaseRobot


class CircularRobot(BaseRobot):
    """
    Base class for circular omnidirectional robots.

    Args:
        **args: see the parent class
        radius: Radius of the robot
        color: Visualization color
        alpha: Visualization alpha
        fill: Visualization fill
        linewidth: Visualization linewidth
        linestyle: Visualization linestyle
        text: Visualization text (visualized in the center of the robot)
        text_color: Visualization text color
        fontsize: Visualization text fontsize
        **kwargs: see the parent class
    """
    def __init__(self, *args, radius: float = 0.5, color: str = "C0", alpha: float = 1.0, 
                 fill: bool = True, linewidth: float = 1.0, linestyle: str = "-",
                 text: str = "", text_color: str = 'white', fontsize: str = None,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.radius = float(radius)

        # visualization parameters
        self.color = color
        self.alpha = alpha
        self.fill = fill
        self.linewidth = linewidth
        self.linestyle = linestyle
        self.text = text
        self.text_color = text_color
        self.fontsize = fontsize
        