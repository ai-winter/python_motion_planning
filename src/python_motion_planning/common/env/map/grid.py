"""
@file: grid.py
@author: Wu Maojia
@update: 2025.12.20
"""
from itertools import product
from typing import Iterable, Union, Tuple, Callable, List, Dict
import time

import numpy as np
from scipy import ndimage

from python_motion_planning.common.env.map.base_map import BaseMap
from python_motion_planning.common.env import Node, TYPES
from python_motion_planning.common.utils.geometry import Geometry


class GridTypeMap:
    """
    Class for Grid Type Map. It is like a np.ndarray, except that its shape and dtype are fixed.

    Args:
        type_map: The np.ndarray type map.

    Examples:
        >>> type_map = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.int8)
        >>> grid_type_map = GridTypeMap(type_map)
        >>> grid_type_map
        GridTypeMap(data=
        [[0 0 0]
         [0 1 0]
         [0 0 0]]
        , shape=(3, 3), dtype=int8)

        >>> grid_type_map.data
        array([[0, 0, 0],
               [0, 1, 0],
               [0, 0, 0]], dtype=int8)

        >>> grid_type_map.shape
        (3, 3)

        >>> grid_type_map.dtype
        dtype('int8')
    """
    def __init__(self, type_map: np.ndarray):
        self._data = np.asarray(type_map)
        self._shape = self._data.shape
        self._dtype = self._data.dtype
        
        self._dtype_options = [np.int8, np.int16, np.int32, np.int64]
        if self._dtype not in self._dtype_options:
            raise ValueError("Dtype must be one of {} instead of {}. If you are not sure, set it to `np.int8`.".format(self._dtype_options, self._dtype))

    def __str__(self) -> str:
        return "GridTypeMap(data=\n{}\n, shape={}, dtype={})".format(self._data, self._shape, self._dtype)

    def __repr__(self) -> str:
        return self.__str__()

    def __getitem__(self, idx):
        return self._data[idx]

    def __setitem__(self, idx, value):
        self._data[idx] = value

    @property
    def data(self) -> np.ndarray:
        return self._data.view()

    @property
    def shape(self) -> Tuple:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype


class Grid(BaseMap):
    """
    Class for Grid Map.
    The shape of each dimension of the grid map is determined by the base world and resolution.
    For each dimension, the conversion equation is: shape_grid = shape_world * resolution + 1
    For example, if the base world is (30, 40) and the resolution is 0.5, the grid map will be (30 * 0.5 + 1, 40 * 0.5 + 1) = (61, 81).

    Args:
        bounds: The size of map in the world (shape: (n, 2) (n>=2)). bounds[i, 0] means the lower bound of the world in the i-th dimension. bounds[i, 1] means the upper bound of the world in the i-th dimension.
        resolution: resolution of the grid map
        type_map: initial type map of the grid map (its shape must be the same as the converted grid map shape, and its dtype must be int)
        inflation_radius: radius of the inflation

    Examples:
        >>> grid_map = Grid(bounds=[[0, 51], [0, 31]], resolution=0.5)
        >>> grid_map
        Grid(bounds=[[ 0. 51.]
         [ 0. 31.]], resolution=0.5)

        >>> grid_map.bounds    # bounds of the base world
        array([[ 0., 51.],
               [ 0., 31.]])

        >>> grid_map.dim
        2

        >>> grid_map.resolution
        0.5

        >>> grid_map.shape   # shape of the grid map
        (102, 62)

        >>> grid_map.dtype
        dtype('int8')

        >>> grid_map.type_map
        GridTypeMap(data=
        [[0 0 0 ... 0 0 0]
         [0 0 0 ... 0 0 0]
         [0 0 0 ... 0 0 0]
         ...
         [0 0 0 ... 0 0 0]
         [0 0 0 ... 0 0 0]
         [0 0 0 ... 0 0 0]]
        , shape=(102, 62), dtype=int8)

        >>> grid_map.map_to_world((1, 2))
        (0.75, 1.25)

        >>> grid_map.world_to_map((0.5, 1.0))
        (0, 2)

        >>> grid_map.get_neighbors(Node((1, 2)))
        [Node((0, 1), (1, 2), 0, 0), Node((0, 2), (1, 2), 0, 0), Node((0, 3), (1, 2), 0, 0), Node((1, 1), (1, 2), 0, 0), Node((1, 3), (1, 2), 0, 0), Node((2, 1), (1, 2), 0, 0), Node((2, 2), (1, 2), 0, 0), Node((2, 3), (1, 2), 0, 0)]

        >>> grid_map.get_neighbors(Node((1, 2)), diagonal=False)
        [Node((2, 2), (1, 2), 0, 0), Node((0, 2), (1, 2), 0, 0), Node((1, 3), (1, 2), 0, 0), Node((1, 1), (1, 2), 0, 0)]

        >>> grid_map[1, 0] = TYPES.OBSTACLE     # place an obstacle
        >>> grid_map.get_neighbors(Node((0, 0)))    # limited within the bounds
        [Node((0, 1), (0, 0), 0, 0), Node((1, 1), (0, 0), 0, 0)]

        >>> grid_map.get_neighbors(Node((grid_map.shape[0] - 1, grid_map.shape[1] - 1)), diagonal=False)  # limited within the boundss
        [Node((100, 61), (101, 61), 0, 0), Node((101, 60), (101, 61), 0, 0)]

        >>> grid_map.line_of_sight((1, 2), (3, 6))
        [(1, 2), (1, 3), (2, 4), (2, 5), (3, 6)]

        >>> grid_map.line_of_sight((1, 2), (1, 2))
        [(1, 2)]

        >>> grid_map.in_collision((1, 2), (3, 6))
        False

        >>> grid_map[1, 3] = TYPES.OBSTACLE
        >>> grid_map.update_esdf()
        >>> grid_map.in_collision((1, 2), (3, 6))
        True
    """
    def __init__(self, 
                bounds: Iterable = [[0, 30], [0, 40]], 
                resolution: float = 1.0, 
                type_map: Union[GridTypeMap, np.ndarray] = None,
                inflation_radius: float = 0.0,
                ) -> None:
        super().__init__(bounds)

        self._resolution = resolution
        shape = tuple([int((self.bounds[i, 1] - self.bounds[i, 0]) / self.resolution) for i in range(self.dim)])

        if type_map is None:
            self.type_map = GridTypeMap(np.zeros(shape, dtype=np.int8))
        else:
            if type_map.shape != shape:
                raise ValueError("Shape must be {} instead of {} with given bounds={} and resolution={}".format(shape, type_map.shape, self.bounds, self.resolution))

            if isinstance(type_map, GridTypeMap):
                self.type_map = type_map
            elif isinstance(type_map, np.ndarray):
                self.type_map = GridTypeMap(type_map)        
            else:
                raise ValueError("Type map must be GridTypeMap or numpy.ndarray instead of {}".format(type(type_map)))

        self._precompute_offsets()
        
        self._esdf = np.zeros(self.shape, dtype=np.float32)
        # self.update_esdf()    # updated in self.inflate_obstacles()

        self.inflation_radius = inflation_radius
        if self.inflation_radius >= 1:
            self.inflate_obstacles(self.inflation_radius)
    
    def __str__(self) -> str:
        return "Grid(bounds={}, resolution={})".format(self.bounds, self.resolution)

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def resolution(self) -> float:
        return self._resolution
    
    @property
    def shape(self) -> tuple:
        return self.type_map.shape
    
    @property
    def dtype(self) -> np.dtype:
        return self.type_map.dtype
    
    @property
    def esdf(self) -> np.ndarray:
        return self._esdf
    
    @property
    def data(self) -> np.ndarray:
        return self.type_map.data
    
    def __getitem__(self, idx):
        return self.type_map[idx]

    def __setitem__(self, idx, value):
        self.type_map[idx] = value

    def map_to_world(self, point: tuple) -> Tuple[float, ...]:
        """
        Convert map coordinates to world coordinates.
        
        Args:
            point: Point in map coordinates.
        
        Returns:
            point: Point in world coordinates.
        """
        if len(point) != self.dim:
            raise ValueError("Point dimension does not match map dimension.")

        return tuple((x + 0.5) * self.resolution + float(self.bounds[i, 0]) for i, x in enumerate(point))

    def world_to_map(self, point: Tuple[float, ...], discrete: bool = True) -> tuple:
        """
        Convert world coordinates to map coordinates.
        
        Args:
            point: Point in world coordinates.
            discrete: Whether to round the coordinates to the nearest integer.
        
        Returns:
            point: Point in map coordinates.
        """
        if len(point) != self.dim:
            raise ValueError("Point dimension does not match map dimension.")
        
        point_map = tuple((x - float(self.bounds[i, 0])) * (1.0 / self.resolution) - 0.5 for i, x in enumerate(point))
        if discrete:
            point_map = self.point_float_to_int(point_map)
        return point_map

    def get_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """
        Get the distance between two points.

        Args:
            p1: Start point.
            p2: Goal point.
        
        Returns:
            dist: Distance between two points.
        """
        return Geometry.dist(p1, p2, type='Euclidean')

    def within_bounds(self, point: Tuple[int, ...]) -> bool:
        """
        Check if a point is within the bounds of the grid map.
        
        Args:
            point: Point to check.
        
        Returns:
            bool: True if the point is within the bounds of the map, False otherwise.
        """
        # if point.dim != self.dim:
        #     raise ValueError("Point dimension does not match map dimension.")

        # return all(0 <= point[i] < self.shape[i] for i in range(self.dim))
        dim = self.dim
        shape = self.shape
        
        for i in range(dim):
            if not (0 <= point[i] < shape[i]):
                return False
        return True

    def is_expandable(self, point: Tuple[int, ...], src_point: Tuple[int, ...] = None) -> bool:
        """
        Check if a point is expandable.
        
        Args:
            point: Point to check.
            src_point: Source point.
        
        Returns:
            expandable: True if the point is expandable, False otherwise.
        """
        if not self.within_bounds(point):
            return False
        if src_point is not None:
            if self.type_map[src_point] == TYPES.INFLATION and self._esdf[point] >= self._esdf[src_point]:
                return True
                
        return not self.type_map[point] == TYPES.OBSTACLE and not self.type_map[point] == TYPES.INFLATION

    def get_neighbors(self, 
                    node: Node, 
                    diagonal: bool = True
                    ) -> list:
        """
        Get neighbor nodes of a given node.
        
        Args:
            node: Node to get neighbor nodes.
            diagonal: Whether to include diagonal neighbors.
        
        Returns:
            nodes: List of neighbor nodes.
        """
        if node.dim != self.dim:
            raise ValueError("Node dimension does not match map dimension.")
        
        offsets = self._diagonal_offsets if diagonal else self._orthogonal_offsets
        
        # Generate all neighbor positions
        # neighbor_positions = current_pos + offsets
        neighbors = [node + offset for offset in offsets]
        filtered_neighbors = []

        for neighbor in neighbors:
            if self.is_expandable(neighbor.current, node.current):
                filtered_neighbors.append(neighbor)
        
        return filtered_neighbors

    def line_of_sight(self, p1: Tuple[int, ...], p2: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """
        N-dimensional line of sight (Bresenham's line algorithm)
        
        Args:
            p1: Start point of the line.
            p2: End point of the line.
        
        Returns:
            points: List of point on the line of sight.
        """
        p1 = np.array(p1)
        p2 = np.array(p2)

        dim = len(p1)
        delta = p2 - p1
        abs_delta = np.abs(delta)
        
        # Determine the main direction axis (the dimension with the greatest change)
        primary_axis = np.argmax(abs_delta)
        primary_step = 1 if delta[primary_axis] > 0 else -1
        
        # Initialize the error variable
        error = np.zeros(dim, dtype=int)
        delta2 = 2 * abs_delta
        
        # Calculate the number of steps and initialize the current point
        steps = abs_delta[primary_axis]
        current = p1
        
        # Allocate the result array
        result = []
        result.append(tuple(int(x) for x in current))
        
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
            
            result.append(tuple(int(x) for x in current))

        return result

    def in_collision(self, p1: Tuple[int, ...], p2: Tuple[int, ...]) -> bool:
        """
        Check if the line of sight between two points is in collision.
        
        Args:
            p1: Start point of the line.
            p2: End point of the line.
        
        Returns:
            in_collision: True if the line of sight is in collision, False otherwise.
        """
        if not self.is_expandable(p1) or not self.is_expandable(p2, p1):
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
        
        for _ in range(steps):
            last_point = current.copy()
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
            if not self.is_expandable(tuple(current), tuple(last_point)):
                return True
        
        return False

    def fill_boundary_with_obstacles(self) -> None:
        """
        Fill the boundary of the map with obstacles.
        """
        for d in range(self.dim):
            # Create a tuple of slice objects to select boundary elements in current dimension
            # First boundary (start index)
            slices_start = [slice(None)] * self.dim
            slices_start[d] = 0
            self.type_map[tuple(slices_start)] = TYPES.OBSTACLE
            
            # Last boundary (end index)
            slices_end = [slice(None)] * self.dim
            slices_end[d] = -1
            self.type_map[tuple(slices_end)] = TYPES.OBSTACLE

    def inflate_obstacles(self, radius: float = 1.0) -> None:
        """
        Inflate the obstacles in the map.
        
        Args:
            radius: Radius of the inflation.
        """
        self.update_esdf()
        mask = (self.esdf <= radius) & (self.type_map.data == TYPES.FREE)
        self.type_map[mask] = TYPES.INFLATION
        self.inflation_radius = radius

    def fill_expands(self, expands: Dict[Tuple[int, ...], Node]) -> None:
        """
        Fill the expands in the map.
        
        Args:
            expands: List of expands.
        """
        for expand in expands.keys():
            if self.type_map[expand] != TYPES.FREE:
                continue
            self.type_map[expand] = TYPES.EXPAND

    def update_esdf(self) -> None:
        """
        Update the ESDF (signed Euclidean Distance Field) based on the obstacles in the map.
        - Obstacle grid ESDF = 0
        - Free grid ESDF > 0. The value is the di/stance to the nearest obstacle
        """
        obstacle_mask = (self.type_map.data == TYPES.OBSTACLE)
        free_mask = ~obstacle_mask

        # distance to obstacles
        dist_outside = ndimage.distance_transform_edt(free_mask, sampling=self.resolution)
        # distance to free space (internal distance of obstacles)
        dist_inside = ndimage.distance_transform_edt(obstacle_mask, sampling=self.resolution)

        self._esdf = dist_outside.astype(np.float32)
        self._esdf[obstacle_mask] = -dist_inside[obstacle_mask]

    def path_map_to_world(self, path: List[tuple]) -> List[Tuple[float, ...]]:
        """
        Convert path from map coordinates to world coordinates

        Args:
            path: a list of map coordinates
        
        Returns:
            path: a list of world coordinates
        """
        return [self.map_to_world(p) for p in path]

    def path_world_to_map(self, path: List[Tuple[float, ...]], discrete: bool = True) -> List[tuple]:
        """
        Convert path from world coordinates to map coordinates

        Args:
            path: a list of world coordinates
            discrete: whether to round the coordinates to the nearest integer
        
        Returns:
            path: a list of map coordinates
        """
        return [self.world_to_map(p, discrete) for p in path]

    def point_float_to_int(self, point: Tuple[float, ...]) -> Tuple[int, ...]:
        """
        Convert a point from float to integer coordinates.

        Args:
            point: a point in float coordinates
        
        Returns:
            point: a point in integer coordinates
        """
        point_int = []
        for d in range(self.dim):
            point_int.append(max(0, min(self.shape[d] - 1, int(round(point[d])))))
        point_int = tuple(point_int)
        return point_int

    def _precompute_offsets(self):
        # Generate all possible offsets (-1, 0, +1) in each dimension
        self._diagonal_offsets = np.array(np.meshgrid(*[[-1, 0, 1]]*self.dim), dtype=self.dtype).T.reshape(-1, self.dim)
        # Remove the zero offset (current node itself)
        self._diagonal_offsets = self._diagonal_offsets[np.any(self._diagonal_offsets != 0, axis=1)]
        # self._diagonal_offsets = [Node((offset.tolist(), dtype=self.dtype)) for offset in self._diagonal_offsets]
        self._diagonal_offsets = [Node(tuple(offset.tolist())) for offset in self._diagonal_offsets]

        # Generate only orthogonal offsets (one dimension changes by ±1)
        self._orthogonal_offsets = np.zeros((2*self.dim, self.dim), dtype=self.dtype)
        for d in range(self.dim):
            self._orthogonal_offsets[2*d, d] = 1
            self._orthogonal_offsets[2*d+1, d] = -1
        # self._orthogonal_offsets = [Node((offset.tolist(), dtype=self.dtype)) for offset in self._orthogonal_offsets]
        self._orthogonal_offsets = [Node(tuple(offset.tolist())) for offset in self._orthogonal_offsets]
