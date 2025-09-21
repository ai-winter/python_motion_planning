import numpy as np
from gymnasium import spaces

class BaseController:
    """
    Controller 基类（与环境解耦）
    - 控制器只知道：observation_space, action_space（由 agent 定义）
    - get_action(obs) -> acceleration vector (dim,)
    """
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Box):
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        """可选：用于在每个 episode 开始时清理内部状态。"""
        pass

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """子类必须实现：返回与 action_space.shape 匹配的 ndarray（加速度）"""
        raise NotImplementedError