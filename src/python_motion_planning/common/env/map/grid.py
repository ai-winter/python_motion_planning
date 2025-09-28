"""
@file: grid.py
@breif: Grid Map for Path Planning
@author: Wu Maojia
@update: 2025.9.5
"""
from itertools import product
from typing import Iterable, Union, Tuple, Callable, List
import time

import numpy as np

from .base_map import BaseMap
from python_motion_planning.common.env import Node, TYPES
from python_motion_planning.common.utils.geometry import Geometry


class GridTypeMap:
    """
    Class for Grid Type Map. It is like a np.ndarray, except that its shape and dtype are fixed.

    Parameters:
        type_map: The np.ndarray type map.

    Examples:
        >>> type_map = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.int8)
        >>> grid_type_map = GridTypeMap(type_map)
        >>> grid_type_map
        GridTypeMap(array(
        [[0 0 0]
         [0 1 0]
         [0 0 0]]
        ), shape=(3, 3), dtype=int8)

        >>> grid_type_map.array
        array([[0, 0, 0],
               [0, 1, 0],
               [0, 0, 0]], dtype=int8)

        >>> grid_type_map.shape
        (3, 3)

        >>> grid_type_map.dtype
        dtype('int8')

        >>> new_array = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=np.int8)

        >>> grid_type_map.update(new_array)

        >>> grid_type_map
        GridTypeMap(array(
        [[1 1 1]
         [0 0 0]
         [0 0 0]]
        ), shape=(3, 3), dtype=int8)
    """
    def __init__(self, type_map: np.ndarray):
        self._array = np.asarray(type_map)
        self._shape = self._array.shape
        self._dtype = self._array.dtype
        
        self._dtype_options = [np.int8, np.int16, np.int32, np.int64]
        if self._dtype not in self._dtype_options:
            raise ValueError("Dtype must be one of {} instead of {}".format(self._dtype_options, self._dtype))

    def __str__(self) -> str:
        return "GridTypeMap(array(\n{}\n), shape={}, dtype={})".format(self._array, self._shape, self._dtype)

    def __repr__(self) -> str:
        return self.__str__()

    def __getitem__(self, idx):
        return self._array[idx]

    def __setitem__(self, idx, value):
        self._array[idx] = value

    @property
    def array(self) -> np.ndarray:
        return self._array.view()

    @property
    def shape(self) -> Tuple:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def update(self, new_array):
        new_array = np.asarray(new_array)
        if new_array.shape != self._shape:
            raise ValueError(f"Shape must be {self._shape}")
        if new_array.dtype != self.dtype:
            raise ValueError(f"New values dtype must be {self.dtype}")
        np.copyto(self._array, new_array)


class Grid(BaseMap):
    """
    Class for Grid Map.
    The shape of each dimension of the grid map is determined by the base world and resolution.
    For each dimension, the conversion equation is: shape_grid = shape_world * resolution + 1
    For example, if the base world is (30, 40) and the resolution is 0.5, the grid map will be (30 * 0.5 + 1, 40 * 0.5 + 1) = (61, 81).

    Parameters:
        bounds: The size of map in the world (shape: (n, 2) (n>=2)). bounds[i, 0] means the lower bound of the world in the i-th dimension. bounds[i, 1] means the upper bound of the world in the i-th dimension.
        resolution: resolution of the grid map
        type_map: initial type map of the grid map (its shape must be the same as the converted grid map shape, and its dtype must be int)
        dtype: data type of coordinates (must be int)

    Examples:
        >>> bounds = [[0, 30], [0, 40]]
        >>> type_map = np.zeros((61, 81), dtype=np.int8)
        >>> grid_map = Grid(bounds=bounds, resolution=0.5, type_map=type_map)
        >>> grid_map
        Grid(bounds=[[ 0. 30.]
         [ 0. 40.]], resolution=0.5)

        >>> grid_map.bounds    # bounds of the base world
        array([[ 0., 30.],
               [ 0., 40.]])

        >>> grid_map.ndim
        2

        >>> grid_map.resolution
        0.5

        >>> grid_map.shape   # shape of the grid map
        (61, 81)

        >>> grid_map.dtype
        <class 'numpy.int32'>

        >>> grid_map.type_map
        GridTypeMap(array(
        [[0 0 0 ... 0 0 0]
         [0 0 0 ... 0 0 0]
         [0 0 0 ... 0 0 0]
         ...
         [0 0 0 ... 0 0 0]
         [0 0 0 ... 0 0 0]
         [0 0 0 ... 0 0 0]]
        ), shape=(61, 81), dtype=int8)

        >>> grid_map.map_to_world((1, 2))
        (0.5, 1.0)

        >>> grid_map.world_to_map((0.5, 1.0))
        (1, 2)

        >>> grid_map.get_neighbors(Node((1, 2)))
        [Node((0, 1), (1, 2), 0, 0), Node((0, 2), (1, 2), 0, 0), Node((0, 3), (1, 2), 0, 0), Node((1, 1), (1, 2), 0, 0), Node((1, 3), (1, 2), 0, 0), Node((2, 1), (1, 2), 0, 0), Node((2, 2), (1, 2), 0, 0), Node((2, 3), (1, 2), 0, 0)]

        >>> grid_map.get_neighbors(Node((1, 2)), diagonal=False)
        [Node((2, 2), (1, 2), 0, 0), Node((0, 2), (1, 2), 0, 0), Node((1, 3), (1, 2), 0, 0), Node((1, 1), (1, 2), 0, 0)]

        >>> grid_map.type_map[1, 0] = TYPES.OBSTACLE     # place an obstacle
        >>> grid_map.get_neighbors(Node((0, 0)))    # limited within the bounds
        [Node((0, 1), (0, 0), 0, 0), Node((1, 1), (0, 0), 0, 0)]

        >>> grid_map.get_neighbors(Node((grid_map.shape[0] - 1, grid_map.shape[1] - 1)), diagonal=False)  # limited within the boundss
        [Node((59, 80), (60, 80), 0, 0), Node((60, 79), (60, 80), 0, 0)]

        >>> grid_map.line_of_sight((1, 2), (3, 6))
        [(1, 2), (1, 3), (2, 4), (2, 5), (3, 6)]

        >>> grid_map.line_of_sight((1, 2), (1, 2))
        [(1, 2)]

        >>> grid_map.in_collision((1, 2), (3, 6))
        False

        >>> grid_map.type_map[1, 3] = TYPES.OBSTACLE
        >>> grid_map.in_collision((1, 2), (3, 6))
        True
    """
    def __init__(self, 
                bounds: Iterable = [[0, 30], [0, 40]], 
                resolution: float = 1.0, 
                type_map: Union[GridTypeMap, np.ndarray] = None, 
                dtype: np.dtype = np.int32
                ) -> None:
        super().__init__(bounds, dtype)

        self._dtype_options = [np.int8, np.int16, np.int32, np.int64]
        if self._dtype not in self._dtype_options:
            raise ValueError("Dtype must be one of {} instead of {}".format(self._dtype_options, self._dtype))

        self._resolution = resolution
        self._shape = tuple([int((self.bounds[i, 1] - self.bounds[i, 0]) / self.resolution) for i in range(self.ndim)])

        if type_map is None:
            self.type_map = GridTypeMap(np.zeros(self._shape, dtype=np.int8))
        else:
            if type_map.shape != self._shape:
                raise ValueError("Shape must be {} instead of {}".format(self._shape, type_map.shape))
            if type_map.dtype not in self._dtype_options:
                raise ValueError("Dtype must be one of {} instead of {}".format(self._dtype_options, type_map.dtype))

            if isinstance(type_map, GridTypeMap):
                self.type_map = type_map
            elif isinstance(type_map, np.ndarray):
                self.type_map = GridTypeMap(type_map)        
            else:
                raise ValueError("Type map must be GridTypeMap or numpy.ndarray instead of {}".format(type(type_map)))

        self._precompute_offsets()
    
    def __str__(self) -> str:
        return "Grid(bounds={}, resolution={})".format(self.bounds, self.resolution)

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def resolution(self) -> float:
        return self._resolution
    
    @property
    def shape(self) -> tuple:
        return self._shape
    
    def map_to_world(self, point: tuple) -> tuple:
        """
        Convert map coordinates to world coordinates.
        
        Parameters:
            point: Point in map coordinates.
        
        Returns:
            point: Point in world coordinates.
        """
        if len(point) != self.ndim:
            raise ValueError("Point dimension does not match map dimension.")

        return tuple((x + 0.5) * self.resolution + float(self.bounds[i, 0]) for i, x in enumerate(point))

    def world_to_map(self, point: tuple) -> tuple:
        """
        Convert world coordinates to map coordinates.
        
        Parameters:
            point: Point in world coordinates.
        
        Returns:
            point: Point in map coordinates.
        """
        if len(point) != self.ndim:
            raise ValueError("Point dimension does not match map dimension.")
        
        return tuple(round((x - float(self.bounds[i, 0])) * (1.0 / self.resolution) - 0.5) for i, x in enumerate(point))

    def get_distance(self, p1: tuple, p2: tuple) -> float:
        """
        Get the distance between two points.

        Parameters:
            p1: Start point.
            p2: Goal point.
        
        Returns:
            dist: Distance between two points.
        """
        return Geometry.dist(p1, p2, type='Euclidean')

    def within_bounds(self, point: tuple) -> bool:
        """
        Check if a point is within the bounds of the grid map.
        
        Parameters:
            point: Point to check.
        
        Returns:
            bool: True if the point is within the bounds of the map, False otherwise.
        """
        # if point.ndim != self.ndim:
        #     raise ValueError("Point dimension does not match map dimension.")

        # return all(0 <= point[i] < self.shape[i] for i in range(self.ndim))
        ndim = self.ndim
        shape = self.shape
        
        for i in range(ndim):
            if not (0 <= point[i] < shape[i]):
                return False
        return True

    def is_expandable(self, point: tuple) -> bool:
        """
        Check if a point is expandable.
        
        Parameters:
            point: Point to check.
        
        Returns:
            expandable: True if the point is expandable, False otherwise.
        """
        return not self.type_map[point] == TYPES.OBSTACLE and not self.type_map[point] == TYPES.INFLATION and self.within_bounds(point)

    def get_neighbors(self, 
                    node: Node, 
                    diagonal: bool = True
                    ) -> list:
        """
        Get neighbor nodes of a given node.
        
        Parameters:
            node: Node to get neighbor nodes.
            diagonal: Whether to include diagonal neighbors.
        
        Returns:
            nodes: List of neighbor nodes.
        """
        if node.ndim != self.ndim:
            raise ValueError("Node dimension does not match map dimension.")

        # current_point = node.current.astype(self.dtype)
        # current_pos = current_point.numpy()
        # neighbors = []
        
        offsets = self._diagonal_offsets if diagonal else self._orthogonal_offsets
        
        # Generate all neighbor positions
        # neighbor_positions = current_pos + offsets
        neighbors = [node + offset for offset in offsets]
        filtered_neighbors = []

        # print(neighbors)

        # Filter out positions outside map bounds
        # for pos in neighbor_positions:
        #     point = (pos, dtype=self.dtype)
        #     if self.within_bounds(point):
        #         if self.type_map[tuple(point)] != TYPES.OBSTACLE:
        #             neighbor_node = Node(point, parent=current_point)
        #             neighbors.append(neighbor_node)
        for neighbor in neighbors:
            if self.is_expandable(neighbor.current):
                filtered_neighbors.append(neighbor)

        # print(filtered_neighbors)
        
        return filtered_neighbors

    def line_of_sight(self, p1: tuple, p2: tuple) -> list:
        """
        N-dimensional line of sight (Bresenham's line algorithm)
        
        Parameters:
            p1: Start point of the line.
            p2: End point of the line.
        
        Returns:
            points: List of point on the line of sight.
        """
        if not self.is_expandable(p1) or not self.is_expandable(p2):
            return []

        p1 = np.array(p1)
        p2 = np.array(p2)

        dim = len(p1)
        delta = p2 - p1
        abs_delta = np.abs(delta)
        
        # Determine the main direction axis (the dimension with the greatest change)
        primary_axis = np.argmax(abs_delta)
        primary_step = 1 if delta[primary_axis] > 0 else -1
        
        # Initialize the error variable
        error = np.zeros(dim, dtype=self.dtype)
        delta2 = 2 * abs_delta
        
        # Calculate the number of steps and initialize the current point
        steps = abs_delta[primary_axis]
        current = p1
        
        # Allocate the result array
        result = []
        result.append(tuple(current))
        
        for i in range(1, steps + 1):
            current[primary_axis] += primary_step
            
            # Update the error for the primary dimension
            for d in range(dim):
                if d == primary_axis:
                    continue
                    
                error[d] += delta2[d]
                if error[d] > abs_delta[primary_axis]:
                    current[d] += 1 if delta[d] > 0 else -1
                    error[d] -= delta2[primary_axis]
            
            result.append(tuple(current))

        return result

    def in_collision(self, p1: tuple, p2: tuple) -> bool:
        """
        Check if the line of sight between two points is in collision.
        
        Parameters:
            p1: Start point of the line.
            p2: End point of the line.
        
        Returns:
            in_collision: True if the line of sight is in collision, False otherwise.
        """
        if not self.is_expandable(p1) or not self.is_expandable(p2):
            return True

        # Corner Case: Start and end points are the same
        if p1 == p2:
            return False
        
        p1 = np.array(p1)
        p2 = np.array(p2)

        # Calculate delta and absolute delta
        delta = p2 - p1
        abs_delta = np.abs(delta)
        
        # Determine the primary axis (the dimension with the greatest change)
        primary_axis = np.argmax(abs_delta)
        primary_step = 1 if delta[primary_axis] > 0 else -1
        
        # Initialize the error variable
        error = np.zeros_like(delta, dtype=np.int32)
        delta2 = 2 * abs_delta
        
        # calculate the number of steps and initialize the current point
        steps = abs_delta[primary_axis]
        current = p1
        
        # Check the start point
        if not self.is_expandable(tuple(current)):
            return True
        
        for _ in range(steps):
            current[primary_axis] += primary_step
            
            # Update the error for the primary dimension
            for d in range(len(delta)):
                if d == primary_axis:
                    continue
                    
                error[d] += delta2[d]
                if error[d] > abs_delta[primary_axis]:
                    current[d] += 1 if delta[d] > 0 else -1
                    error[d] -= delta2[primary_axis]
            
            # Check the current point
            if not self.is_expandable(tuple(current)):
                return True
        
        return False

    def fill_boundary_with_obstacles(self) -> None:
        """
        Fill the boundary of the map with obstacles.
        """
        self.type_map[0, :] = TYPES.OBSTACLE
        self.type_map[-1, :] = TYPES.OBSTACLE
        self.type_map[:, 0] = TYPES.OBSTACLE
        self.type_map[:, -1] = TYPES.OBSTACLE

    def inflate_obstacles(self, radius: float = 1.0) -> None:
        """
        Inflate the obstacles in the map.
        
        Parameters:
            radius: Radius of the inflation.
        """
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if self.type_map[i, j] == TYPES.OBSTACLE:
                    for k in range(round(i-radius), round(i+radius+1)):
                        for l in range(round(j-radius), round(j+radius+1)):
                            if k < 0 or k >= self.shape[0] or l < 0 or l >= self.shape[1]:
                                continue
                            if self.type_map[k, l] == TYPES.FREE and (k - i)**2 + (l - j)**2 <= radius**2:
                                self.type_map[k, l] = TYPES.INFLATION

    def fill_expands(self, expands: List[Node]) -> None:
        """
        Fill the expands in the map.
        
        Parameters:
            expands: List of expands.
        """
        for expand in expands:
            if self.type_map[expand.current] != TYPES.FREE:
                continue
            self.type_map[expand.current] = TYPES.EXPAND

    def _precompute_offsets(self):
        # Generate all possible offsets (-1, 0, +1) in each dimension
        self._diagonal_offsets = np.array(np.meshgrid(*[[-1, 0, 1]]*self.ndim), dtype=self.dtype).T.reshape(-1, self.ndim)
        # Remove the zero offset (current node itself)
        self._diagonal_offsets = self._diagonal_offsets[np.any(self._diagonal_offsets != 0, axis=1)]
        # self._diagonal_offsets = [Node((offset.tolist(), dtype=self.dtype)) for offset in self._diagonal_offsets]
        self._diagonal_offsets = [Node(tuple(offset.tolist())) for offset in self._diagonal_offsets]

        # Generate only orthogonal offsets (one dimension changes by ±1)
        self._orthogonal_offsets = np.zeros((2*self.ndim, self.ndim), dtype=self.dtype)
        for dim in range(self.ndim):
            self._orthogonal_offsets[2*dim, dim] = 1
            self._orthogonal_offsets[2*dim+1, dim] = -1
        # self._orthogonal_offsets = [Node((offset.tolist(), dtype=self.dtype)) for offset in self._orthogonal_offsets]
        self._orthogonal_offsets = [Node(tuple(offset.tolist())) for offset in self._orthogonal_offsets]
