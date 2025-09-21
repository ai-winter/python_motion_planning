from typing import List, Tuple

import numpy as np
from gymnasium import spaces

class BaseController:
    """
    Base class for controllers.
    - The controller only knows observation_space, action_space

    Parameters:
        observation_space: observation space ([pos, vel, rel_pos_robot1, rel_pos_robot2, ...], each sub-vector length=dim)
        action_space: action space ([acc], length=dim)
        path: path to follow
        dt: time step for control
        max_speed: maximum speed of the robot
    """
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Box, path: List[Tuple[float, ...]],
                 dt: float, max_speed: float = np.inf):
        self.observation_space = observation_space
        self.action_space = action_space
        self.path = path
        self.dt = dt
        self.max_speed = max_speed
        self.goal = path[-1] if len(path) > 0 else None

    def reset(self):
        """
        Reset the controller to initial state.
        """
        pass

    def get_action(self, obs: np.ndarray) -> Tuple[np.ndarray, tuple]:
        """
        Get action from observation.

        Parameters:
            obs: observation ([pos, vel, rel_pos_robot1, rel_pos_robot2, ...], each sub-vector length=dim)

        Returns:
            action: action ([acc], length=dim)
            target: lookahead point
        """
        return np.zeros(self.action_space.shape), self.goal

    def clip_velocity(self, v: np.ndarray) -> np.ndarray:
        """
        Clip the velocity to the maximum allowed value.

        Parameters:
            v (np.ndarray): The velocity to clip.

        Returns:
            np.ndarray: The clipped velocity.
        """
        return v if np.linalg.norm(v) <= self.max_speed else v / np.linalg.norm(v) * self.max_speed

    def clip_action(self, a: np.ndarray) -> np.ndarray:
        """
        Clip action to action bounds.

        Parameters:
            a(np.ndarray): Action vector

        Returns:
            np.ndarray: Clipped action vector
        """
        return np.clip(a, self.action_space.low, self.action_space.high)