"""
@file: grid.py
@breif: Grid Map for Path Planning
@author: Wu Maojia
@update: 2025.3.29
"""
from itertools import product
from typing import Iterable, Union, Tuple, Callable
import time

import numpy as np

from python_motion_planning.common.env import World, Node, TYPES
from python_motion_planning.common.geometry.point import *
from python_motion_planning.common.env.map import Map


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
        self._array = np.array(type_map)
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
        return self._array.copy()

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


class Grid(Map):
    """
    Class for Grid Map.
    The shape of each dimension of the grid map is determined by the base world and resolution.
    For each dimension, the conversion equation is: shape_grid = shape_world * resolution + 1
    For example, if the base world is (30, 40) and the resolution is 0.5, the grid map will be (30 * 0.5 + 1, 40 * 0.5 + 1) = (61, 81).

    Parameters:
        world: Base world.
        type_map: initial type map of the grid map (its shape must be the same as the converted grid map shape, and its dtype must be int)
        resolution: resolution of the grid map
        dtype: data type of coordinates (must be int)

    Examples:
        >>> type_map = np.zeros((61, 81), dtype=np.int8)
        >>> grid_map = Grid(world=(30, 40), type_map=type_map, resolution=0.5)
        >>> grid_map
        Grid(World((30, 40)), resolution=0.5)

        >>> grid_map.world
        World((30, 40))

        >>> grid_map.bounds    # bounds of the base world
        (30, 40)

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

        >>> grid_map.mapToWorld(Point2D(1, 2))
        Point2D([0.5, 1.0], dtype=float64)

        >>> grid_map.worldToMap(Point2D(0.5, 1.0))
        Point2D([1, 2], dtype=int32)

        >>> grid_map.getNeighbor(Node(Point2D(1, 2)))
        [Node(PointND([0, 1], dtype=int32), Point2D([1, 2], dtype=int32), 1.4142135623730951, 0), Node(PointND([0, 2], dtype=int32), Point2D([1, 2], dtype=int32), 1.0, 0), Node(PointND([0, 3], dtype=int32), Point2D([1, 2], dtype=int32), 1.4142135623730951, 0), Node(PointND([1, 1], dtype=int32), Point2D([1, 2], dtype=int32), 1.0, 0), Node(PointND([1, 3], dtype=int32), Point2D([1, 2], dtype=int32), 1.0, 0), Node(PointND([2, 1], dtype=int32), Point2D([1, 2], dtype=int32), 1.4142135623730951, 0), Node(PointND([2, 2], dtype=int32), Point2D([1, 2], dtype=int32), 1.0, 0), Node(PointND([2, 3], dtype=int32), Point2D([1, 2], dtype=int32), 1.4142135623730951, 0)]
        
        >>> grid_map.getNeighbor(Node(Point2D(1, 2)), diagonal=False)
        [Node(PointND([2, 2], dtype=int32), Point2D([1, 2], dtype=int32), 1.0, 0), Node(PointND([0, 2], dtype=int32), Point2D([1, 2], dtype=int32), 1.0, 0), Node(PointND([1, 3], dtype=int32), Point2D([1, 2], dtype=int32), 1.0, 0), Node(PointND([1, 1], dtype=int32), Point2D([1, 2], dtype=int32), 1.0, 0)]

        >>> grid_map.getNeighbor(Node(Point2D(0, 0)))    # limited within the bounds
        [Node(PointND([0, 1], dtype=int32), Point2D([0, 0], dtype=int32), 1.0, 0), Node(PointND([1, 0], dtype=int32), Point2D([0, 0], dtype=int32), 1.0, 0), Node(PointND([1, 1], dtype=int32), Point2D([0, 0], dtype=int32), 1.4142135623730951, 0)]

        >>> grid_map.getNeighbor(Node(Point2D(grid_map.shape[0] - 1, grid_map.shape[1] - 1)), diagonal=False)  # limited within the boundss
        [Node(PointND([59, 80], dtype=int32), Point2D([60, 80], dtype=int32), 1.0, 0), Node(PointND([60, 79], dtype=int32), Point2D([60, 80], dtype=int32), 1.0, 0)]

        >>> grid_map.lineOfSight(Point2D(1, 2), Point2D(3, 6))
        array([[1, 2],
               [1, 3],
               [2, 4],
               [2, 5],
               [3, 6]], dtype=int32)

        >>> grid_map.lineOfSight(Point2D(1, 2), Point2D(1, 2))
        array([[1, 2]], dtype=int32)

        >>> grid_map.inCollision(Point2D(1, 2), Point2D(3, 6))
        False

        >>> grid_map.type_map[1, 3] = TYPES.OBSTACLE
        >>> grid_map.inCollision(Point2D(1, 2), Point2D(3, 6))
        True
    """
    def __init__(self, 
                world: Union[World, Iterable], 
                type_map: Union[GridTypeMap, np.ndarray] = None, 
                resolution: float = 1.0, 
                dtype: np.dtype = np.int32
                ) -> None:
        super().__init__(world, dtype)
        
        self._dtype_options = [np.int8, np.int16, np.int32, np.int64]
        if self._dtype not in self._dtype_options:
            raise ValueError("Dtype must be one of {} instead of {}".format(self._dtype_options, self._dtype))

        self._resolution = resolution
        self._shape = tuple([int(self.world.bounds[i] / self.resolution) + 1 for i in range(self.ndim)])

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
    
    def __str__(self) -> str:
        return "Grid({}, resolution={})".format(self.world, self.resolution)

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def resolution(self) -> float:
        return self._resolution
    
    @property
    def shape(self) -> tuple:
        return self._shape
    
    def mapToWorld(self, point: PointND) -> PointND:
        """
        Convert map coordinates to world coordinates.
        
        Parameters:
            point: Point in map coordinates.
        
        Returns:
            point: Point in world coordinates.
        """
        if point.ndim != self.ndim:
            raise ValueError("Point dimension does not match map dimension.")
        
        return point.astype(self.world.dtype) * self.resolution

    def worldToMap(self, point: PointND) -> PointND:
        """
        Convert world coordinates to map coordinates.
        
        Parameters:
            point: Point in world coordinates.
        
        Returns:
            point: Point in map coordinates.
        """
        if point.ndim != self.ndim:
            raise ValueError("Point dimension does not match map dimension.")
        
        return (point * (1.0 / self.resolution)).astype(self.dtype)

    def withinBounds(self, point: PointND) -> bool:
        """
        Check if a point is within the bounds of the grid map.
        
        Parameters:
            point: Point to check.
        
        Returns:
            bool: True if the point is within the bounds of the map, False otherwise.
        """
        if point.ndim != self.ndim:
            raise ValueError("Point dimension does not match map dimension.")

        return all(0 <= point[i] < self.shape[i] for i in range(self.ndim))

    def getDistance(self, p1: PointND, p2: PointND) -> float:
        """
        Get the distance between two points.

        Parameters:
            p1: Start point.
            p2: Goal point.
        
        Returns:
            dist: Distance between two points.
        """
        return p1.dist(p2, type='Euclidean')

    def getNeighbor(self, 
                    node: Node, 
                    diagonal: bool = True, 
                    cost_function: Callable[[PointND, PointND], float] = None,
                    heuristic_function: Callable[[PointND], float] = None 
                    ) -> list:
        """
        Get neighbor nodes of a given node.
        
        Parameters:
            node: Node to get neighbor nodes.
            diagonal: Whether to include diagonal neighbors.
            cost_function: Cost function to calculate the cost between two points (default: getDistance(p1, p2)).
            heuristic_function: Heuristic function to calculate the heuristic value of a node (default: return 0).
        
        Returns:
            nodes: List of neighbor nodes.
        """
        if node.ndim != self.ndim:
            raise ValueError("Node dimension does not match map dimension.")

        current_point = node.current.astype(self.dtype)
        current_pos = np.array(current_point)
        neighbors = []
        
        if diagonal:
            # Generate all possible offsets (-1, 0, +1) in each dimension
            offsets = np.array(np.meshgrid(*[[-1, 0, 1]]*self.ndim), dtype=self.dtype).T.reshape(-1, self.ndim)
            # Remove the zero offset (current node itself)
            offsets = offsets[np.any(offsets != 0, axis=1)]
        else:
            # Generate only orthogonal offsets (one dimension changes by ±1)
            offsets = np.zeros((2*self.ndim, self.ndim), dtype=self.dtype)
            for dim in range(self.ndim):
                offsets[2*dim, dim] = 1
                offsets[2*dim+1, dim] = -1
        
        # Generate all neighbor positions
        neighbor_positions = current_pos + offsets

        if cost_function is None:
            cost_function = self.getDistance

        if heuristic_function is None:
            heuristic_function = lambda p: 0

        # Filter out positions outside map bounds
        for pos in neighbor_positions:
            point = PointND(pos, dtype=self.dtype)
            if self.withinBounds(point):
                neighbor_node = Node(point, parent=current_point, g=node.g + cost_function(current_point, point), h=heuristic_function(point))
                neighbors.append(neighbor_node)
        
        return neighbors
        
    def lineOfSight(self, p1: PointND, p2: PointND) -> list:
        """
        N-dimensional line of sight (Bresenham's line algorithm)
        
        Parameters:
        
        Returns:
        """
        if not self.withinBounds(p1) or not self.withinBounds(p2):
            return []

        p1 = np.array(p1, dtype=self.dtype)
        p2 = np.array(p2, dtype=self.dtype)

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
        current = p1.copy()
        
        # Allocate the result array
        result = np.zeros((steps + 1, dim), dtype=self.dtype)
        result[0] = current
        
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
            
            result[i] = current

        return result

    def inCollision(self, p1: PointND, p2: PointND) -> bool:
        """
        Check if the line of sight between two points is in collision.
        
        Parameters:
            p1: Start point of the line.
            p2: End point of the line.
        
        Returns:
            in_collision: True if the line of sight is in collision, False otherwise.
        """
        if not self.withinBounds(p1) or not self.withinBounds(p2):
            return True

        p1 = np.array(p1, dtype=np.int32)
        p2 = np.array(p2, dtype=np.int32)
        
        # Corner Case: Start and end points are the same
        if np.all(p1 == p2):
            return self.type_map[tuple(p1)] == TYPES.OBSTACLE
        
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
        current = p1.copy()
        
        # Check the start point
        if self.type_map[tuple(current)] == TYPES.OBSTACLE:
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
            if self.type_map[tuple(current)] == TYPES.OBSTACLE:
                return True
        
        return False
            