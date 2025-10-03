"""
@file: random_controller.py
@author: Wu Maojia
@update: 2025.10.3
"""
from typing import Tuple

import numpy as np

from .base_controller import BaseController

class RandomController(BaseController):
    """
    Random controller
    """
    def get_action(self, obs: np.ndarray) -> Tuple[np.ndarray, tuple]:
        """
        Randomly sample action in action space.

        Parameters:
            obs: observation ([pos, vel, rel_pos_robot1, rel_pos_robot2, ...], each sub-vector length=dim)

        Returns:
            action: action ([acc], length=dim)
            target: lookahead point
        """
        return self.action_space.sample(), self.goal