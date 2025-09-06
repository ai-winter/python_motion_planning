"""
@file: world.py
@breif: Motion Planning Base World
@author: Wu Maojia
@update: 2025.3.29
"""
from typing import Iterable, Union
from abc import ABC, abstractmethod

import numpy as np
# import gym
# from gym import spaces

# class MultiAgentPathEnv(gym.Env):
#     """
#     Continuous multi-agent path planning environment.
#     Supports n-dimensional space (2D, 3D, etc.).
#     """
#     metadata = {'render.modes': ['human']}
    
#     def __init__(self, 
#                  dimension: int = 2, 
#                  world_size: float = 10.0,
#                  max_agents: int = 10,
#                  max_steps: int = 500):
#         """
#         Initialize the environment.
        
#         Args:
#             dimension: Dimensionality of the space (2 for 2D, 3 for 3D, etc.)
#             world_size: Size of the world (same for all dimensions)
#             max_agents: Maximum number of agents allowed in the environment
#             max_steps: Maximum episode length
#         """
#         super(MultiAgentPathEnv, self).__init__()
        
#         self.dimension = dimension
#         self.world_size = world_size
#         self.max_agents = max_agents
#         self.max_steps = max_steps
#         self.current_step = 0
        
#         # Space boundaries (same for all dimensions)
#         self.low = -world_size/2
#         self.high = world_size/2
        
#         # Agents in the environment
#         self.agents: List[Agent] = []
        
#         # Define action and observation spaces
#         # Action space: continuous movement vector for each agent
#         self.action_space = spaces.Box(
#             low=-1.0, high=1.0, 
#             shape=(max_agents, dimension), 
#             dtype=np.float32
#         )
        
#         # Observation space: positions and goals for all agents
#         # We need to create Box spaces with proper shapes
#         self.observation_space = spaces.Dict({
#             'positions': spaces.Box(
#                 low=self.low, high=self.high,
#                 shape=(max_agents, dimension),
#                 dtype=np.float32
#             ),
#             'goals': spaces.Box(
#                 low=self.low, high=self.high,
#                 shape=(max_agents, dimension),
#                 dtype=np.float32
#             ),
#             'agent_ids': spaces.Box(
#                 low=0, high=max_agents-1,
#                 shape=(max_agents,),
#                 dtype=np.int32
#             )
#         })
        
#     def add_agent(self, agent: Agent):
#         """
#         Add an agent to the environment.
        
#         Args:
#             agent: Agent object to add
#         """
#         if len(self.agents) >= self.max_agents:
#             raise ValueError(f"Cannot add more than {self.max_agents} agents")
            
#         # Ensure agent position is within bounds
#         agent.position = np.clip(agent.position, self.low, self.high)
#         self.agents.append(agent)
        
#     def reset(self) -> Dict:
#         """
#         Reset the environment to initial state.
        
#         Returns:
#             observation: Initial observation of all agents
#         """
#         self.current_step = 0
        
#         # Reset agent positions (could be randomized in actual implementation)
#         for agent in self.agents:
#             # This would be customized based on your needs
#             agent.position = np.random.uniform(
#                 self.low, self.high, size=self.dimension
#             )
            
#         return self._get_observation()
    
#     def step(self, actions: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
#         """
#         Execute one time step in the environment.
        
#         Args:
#             actions: Array of actions for each agent (shape: [num_agents, dimension])
            
#         Returns:
#             observation: New observation after taking actions
#             reward: Cumulative reward for this step
#             done: Whether episode has finished
#             info: Additional information
#         """
#         self.current_step += 1
        
#         # Apply actions to each agent
#         for i, agent in enumerate(self.agents):
#             if i < actions.shape[0]:  # Ensure we don't exceed provided actions
#                 agent.apply_action(actions[i])
                
#                 # Keep agent within world bounds
#                 agent.position = np.clip(agent.position, self.low, self.high)
        
#         # Get new observation
#         obs = self._get_observation()
        
#         # Calculate rewards (placeholder - implement your own reward function)
#         rewards = self._calculate_rewards()
#         total_reward = sum(rewards.values())
        
#         # Check termination conditions
#         done = self.current_step >= self.max_steps or self._all_goals_reached()
        
#         # Additional info
#         info = {
#             'rewards': rewards,
#             'collisions': self._check_collisions(),
#             'success': self._all_goals_reached()
#         }
        
#         return obs, total_reward, done, info
    
#     def _get_observation(self) -> Dict:
#         """
#         Get current observation of all agents.
        
#         Returns:
#             Dictionary containing positions, goals, and IDs of all agents
#         """
#         positions = np.full((self.max_agents, self.dimension), self.low, dtype=np.float32)
#         goals = np.full((self.max_agents, self.dimension), self.low, dtype=np.float32)
#         ids = np.zeros(self.max_agents, dtype=np.int32)
        
#         for i, agent in enumerate(self.agents):
#             positions[i] = agent.position
#             goals[i] = agent.goal if agent.goal is not None else agent.position
#             ids[i] = agent.id
            
#         return {
#             'positions': positions,
#             'goals': goals,
#             'agent_ids': ids
#         }
    
#     def _calculate_rewards(self) -> Dict[int, float]:
#         """
#         Calculate rewards for each agent.
#         Placeholder implementation - should be customized.
        
#         Returns:
#             Dictionary mapping agent IDs to their individual rewards
#         """
#         rewards = {}
#         for agent in self.agents:
#             # Simple reward: negative distance to goal
#             if agent.goal is not None:
#                 dist = np.linalg.norm(agent.position - agent.goal)
#                 rewards[agent.id] = -dist
#             else:
#                 rewards[agent.id] = 0.0
                
#             # Add collision penalty if needed
#             if self._check_collisions().get(agent.id, False):
#                 rewards[agent.id] -= 10.0
                
#         return rewards
    
#     def _check_collisions(self) -> Dict[int, bool]:
#         """
#         Check for collisions between agents.
        
#         Returns:
#             Dictionary mapping agent IDs to collision status (True if colliding)
#         """
#         collisions = {agent.id: False for agent in self.agents}
        
#         # Check all pairs of agents
#         for i, a1 in enumerate(self.agents):
#             for j, a2 in enumerate(self.agents):
#                 if i < j:  # Avoid duplicate checks
#                     dist = np.linalg.norm(a1.position - a2.position)
#                     if dist < (a1.radius + a2.radius):
#                         collisions[a1.id] = True
#                         collisions[a2.id] = True
                        
#         return collisions
    
#     def _all_goals_reached(self) -> bool:
#         """
#         Check if all agents have reached their goals.
        
#         Returns:
#             True if all agents are at their goals, False otherwise
#         """
#         for agent in self.agents:
#             if agent.goal is None:
#                 continue
#             if np.linalg.norm(agent.position - agent.goal) > agent.radius:
#                 return False
#         return True
    
#     def render(self, mode='human'):
#         """Render the environment (placeholder - implement visualization)."""
#         if mode == 'human':
#             print(f"Step: {self.current_step}")
#             for agent in self.agents:
#                 print(f"Agent {agent.id}: Position {agent.position}, Goal {agent.goal}")
                
#     def close(self):
#         """Clean up resources."""
#         pass


class World(ABC):
    """
    Class for Motion Planning Base World. It is continuous and in n-d Cartesian coordinate system.

    Parameters:
        bounds: boundaries of world (length of boundaries means the number of dimensions)
        dtype: data type of coordinates (must be float)

    Examples:
        >>> world = World((30, 40))

        >>> world
        World((30, 40))

        >>> world.bounds
        (30, 40)

        >>> world.ndim
        2

        >>> world.dtype
        <class 'numpy.float64'>
    """
    def __init__(self, bounds: Iterable, dtype: np.dtype = np.float64) -> None:
        super().__init__()
        try:
            self._bounds = tuple(bounds)
            self._ndim = len(self._bounds)
            self._dtype = dtype

            if self._ndim <= 1:
                raise ValueError("Input length must be greater than 1.")
            if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in self._bounds):
                raise ValueError("Input must be a non-empty 1D array.")

            self._dtype_options = [np.float64, np.float32, np.float16]
            if self._dtype not in self._dtype_options:
                raise ValueError("Dtype must be one of {} instead of {}".format(self._dtype_options, self._dtype))

        except Exception as e:
            raise ValueError("Invalid input for World: {}".format(e))

    def __str__(self) -> str:
        return "World({})".format(self._bounds)

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def bounds(self) -> tuple:
        return self._bounds

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def reset(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError
        


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)