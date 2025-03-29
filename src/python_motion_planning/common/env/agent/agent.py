"""
@file: agent.py
@breif: Agent class for the environment.
@author: Wu Maojia
@update: 2025.3.29
"""
from typing import Iterable, Union
from abc import ABC, abstractmethod

import numpy as np

class Agent(ABC):
    """
    Base class for agents in the environment.
    Agents should be defined externally and passed to the environment.
    """
    def __init__(self, 
                agent_id: int, 
                init_pos: Iterable,
                action_space,
                observation_space,
                policy: Callable,
                ):
        """
        Initialize an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            initial_position: Starting position of the agent (n-dimensional array)
        """
        super().__init__()
        self.id = agent_id
        self.position = np.array(init_pos, dtype=np.float64)
        self.policy = policy

    def step(self):
        pass
        
    # def set_goal(self, goal_position: np.ndarray):
    #     """Set the target goal position for the agent."""
    #     self.goal = goal_position.copy()
        
    # def get_observation(self) -> Dict:
    #     """Get the agent's current observation (position, goal, etc.)"""
    #     return {
    #         'position': self.position,
    #         'goal': self.goal,
    #         'id': self.id
    #     }
        
    # def apply_action(self, action: np.ndarray, dt: float = 0.1):
    #     """
    #     Apply movement action to the agent.
        
    #     Args:
    #         action: Movement vector (normalized direction * speed)
    #         dt: Time step size (for scaling movement)
    #     """
    #     # Clip action to max speed
    #     speed = np.linalg.norm(action)
    #     if speed > self.max_speed:
    #         action = action / speed * self.max_speed
            
    #     # Update position
    #     self.position += action * dt