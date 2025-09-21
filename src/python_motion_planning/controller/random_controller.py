import numpy as np

from .base_controller import BaseController

class RandomController(BaseController):
    """默认随机控制器：在 action_space 内均匀采样。"""
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        return self.action_space.sample(), self.goal