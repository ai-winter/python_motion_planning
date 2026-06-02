"""
@file: grid.py
@author: Wu Maojia
@update: 2026.6.2
"""
from itertools import product
from typing import Iterable, Union, Tuple, Callable, List, Dict
import math
import time

import numpy as np
from scipy import ndimage

try:
    from numba import njit as _numba_njit
except Exception:  # pragma: no cover - keeps import compatibility without numba.
    _numba_njit = None

from python_motion_planning.common.env.map.base_map import BaseMap
from python_motion_planning.common.env import Node, TYPES
from python_motion_planning.common.utils.geometry import Geometry


def _njit(*args, **kwargs):
    if _numba_njit is None:
        if args and callable(args[0]):
            return args[0]

        def decorator(func):
            return func

        return decorator

    return _numba_njit(*args, **kwargs)


@_njit(cache=True)
def _grid_flat_index(point: np.ndarray, shape: np.ndarray) -> int:
    idx = 0
    for d in range(shape.size):
        idx = idx * shape[d] + point[d]
    return idx


@_njit(cache=True)
def _grid_within_bounds(point: np.ndarray, shape: np.ndarray) -> bool:
    for d in range(shape.size):
        if point[d] < 0 or point[d] >= shape[d]:
            return False
    return True


@_njit(cache=True)
def _grid_is_expandable(
    point: np.ndarray,
    src_point: np.ndarray,
    has_src_point: bool,
    shape: np.ndarray,
    type_map: np.ndarray,
    esdf: np.ndarray,
    obstacle_type: int,
    inflation_type: int,
) -> bool:
    if not _grid_within_bounds(point, shape):
        return False

    point_idx = _grid_flat_index(point, shape)
    if has_src_point:
        src_idx = _grid_flat_index(src_point, shape)
        if type_map[src_idx] == inflation_type and esdf[point_idx] >= esdf[src_idx]:
            return True

    point_type = type_map[point_idx]
    return point_type != obstacle_type and point_type != inflation_type


@_njit(cache=True)
def _grid_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    dist_square = 0.0
    for d in range(p1.size):
        diff = p1[d] - p2[d]
        dist_square += diff * diff
    return math.sqrt(dist_square)


@_njit(cache=True)
def _grid_map_to_world(point: np.ndarray, bounds: np.ndarray, resolution: float) -> np.ndarray:
    point_world = np.empty(point.size, dtype=np.float64)
    for d in range(point.size):
        point_world[d] = (point[d] + 0.5) * resolution + bounds[d, 0]
    return point_world


@_njit(cache=True)
def _grid_point_float_to_int(point: np.ndarray, shape: np.ndarray) -> np.ndarray:
    point_int = np.empty(shape.size, dtype=np.int64)
    for d in range(shape.size):
        value = int(round(point[d]))
        if value < 0:
            value = 0
        elif value >= shape[d]:
            value = shape[d] - 1
        point_int[d] = value
    return point_int


@_njit(cache=True)
def _grid_world_to_map_float(point: np.ndarray, bounds: np.ndarray, resolution: float) -> np.ndarray:
    point_map = np.empty(point.size, dtype=np.float64)
    inv_resolution = 1.0 / resolution
    for d in range(point.size):
        point_map[d] = (point[d] - bounds[d, 0]) * inv_resolution - 0.5
    return point_map


@_njit(cache=True)
def _grid_world_to_map_int(
    point: np.ndarray,
    bounds: np.ndarray,
    resolution: float,
    shape: np.ndarray,
) -> np.ndarray:
    point_map = np.empty(shape.size, dtype=np.int64)
    inv_resolution = 1.0 / resolution
    for d in range(shape.size):
        value = int(round((point[d] - bounds[d, 0]) * inv_resolution - 0.5))
        if value < 0:
            value = 0
        elif value >= shape[d]:
            value = shape[d] - 1
        point_map[d] = value
    return point_map


@_njit(cache=True)
def _grid_line_of_sight(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    dim = p1.size
    delta = np.empty(dim, dtype=np.int64)
    abs_delta = np.empty(dim, dtype=np.int64)
    delta2 = np.empty(dim, dtype=np.int64)

    primary_axis = 0
    max_delta = 0
    for d in range(dim):
        delta[d] = p2[d] - p1[d]
        abs_delta[d] = abs(delta[d])
        delta2[d] = 2 * abs_delta[d]
        if abs_delta[d] > max_delta:
            max_delta = abs_delta[d]
            primary_axis = d

    primary_step = 1 if delta[primary_axis] > 0 else -1
    steps = abs_delta[primary_axis]
    result = np.empty((steps + 1, dim), dtype=np.int64)
    current = p1.copy()

    for d in range(dim):
        result[0, d] = current[d]

    error = np.zeros(dim, dtype=np.int64)
    for i in range(1, steps + 1):
        current[primary_axis] += primary_step

        for d in range(dim):
            if d == primary_axis:
                continue

            error[d] += delta2[d]
            if error[d] > abs_delta[primary_axis]:
                current[d] += 1 if delta[d] > 0 else -1
                error[d] -= delta2[primary_axis]

        for d in range(dim):
            result[i, d] = current[d]

    return result


@_njit(cache=True)
def _grid_in_collision(
    p1: np.ndarray,
    p2: np.ndarray,
    shape: np.ndarray,
    type_map: np.ndarray,
    esdf: np.ndarray,
    obstacle_type: int,
    inflation_type: int,
) -> bool:
    if not _grid_is_expandable(p1, p1, False, shape, type_map, esdf, obstacle_type, inflation_type):
        return True
    if not _grid_is_expandable(p2, p1, True, shape, type_map, esdf, obstacle_type, inflation_type):
        return True

    dim = p1.size
    same_point = True
    for d in range(dim):
        if p1[d] != p2[d]:
            same_point = False
            break
    if same_point:
        return False

    delta = np.empty(dim, dtype=np.int64)
    abs_delta = np.empty(dim, dtype=np.int64)
    delta2 = np.empty(dim, dtype=np.int64)

    primary_axis = 0
    max_delta = 0
    for d in range(dim):
        delta[d] = p2[d] - p1[d]
        abs_delta[d] = abs(delta[d])
        delta2[d] = 2 * abs_delta[d]
        if abs_delta[d] > max_delta:
            max_delta = abs_delta[d]
            primary_axis = d

    primary_step = 1 if delta[primary_axis] > 0 else -1
    steps = abs_delta[primary_axis]
    current = p1.copy()
    last_point = np.empty(dim, dtype=np.int64)
    error = np.zeros(dim, dtype=np.int64)

    for _ in range(steps):
        for d in range(dim):
            last_point[d] = current[d]

        current[primary_axis] += primary_step

        for d in range(dim):
            if d == primary_axis:
                continue

            error[d] += delta2[d]
            if error[d] > abs_delta[primary_axis]:
                current[d] += 1 if delta[d] > 0 else -1
                error[d] -= delta2[primary_axis]

        if not _grid_is_expandable(current, last_point, True, shape, type_map, esdf, obstacle_type, inflation_type):
            return True

    return False


@_njit(cache=True)
def _grid_neighbor_positions_and_mask(
    current: np.ndarray,
    offsets: np.ndarray,
    shape: np.ndarray,
    type_map: np.ndarray,
    esdf: np.ndarray,
    obstacle_type: int,
    inflation_type: int,
) -> Tuple[np.ndarray, np.ndarray]:
    node_num = offsets.shape[0]
    dim = offsets.shape[1]
    positions = np.empty((node_num, dim), dtype=np.int64)
    mask = np.zeros(node_num, dtype=np.bool_)
    neighbor = np.empty(dim, dtype=np.int64)

    for i in range(node_num):
        for d in range(dim):
            neighbor[d] = current[d] + offsets[i, d]
            positions[i, d] = neighbor[d]

        mask[i] = _grid_is_expandable(neighbor, current, True, shape, type_map, esdf, obstacle_type, inflation_type)

    return positions, mask


@_njit(cache=True)
def _grid_path_map_to_world(points: np.ndarray, bounds: np.ndarray, resolution: float) -> np.ndarray:
    path_world = np.empty((points.shape[0], points.shape[1]), dtype=np.float64)
    for i in range(points.shape[0]):
        for d in range(points.shape[1]):
            path_world[i, d] = (points[i, d] + 0.5) * resolution + bounds[d, 0]
    return path_world


@_njit(cache=True)
def _grid_path_world_to_map_float(points: np.ndarray, bounds: np.ndarray, resolution: float) -> np.ndarray:
    path_map = np.empty((points.shape[0], points.shape[1]), dtype=np.float64)
    inv_resolution = 1.0 / resolution
    for i in range(points.shape[0]):
        for d in range(points.shape[1]):
            path_map[i, d] = (points[i, d] - bounds[d, 0]) * inv_resolution - 0.5
    return path_map


@_njit(cache=True)
def _grid_path_world_to_map_int(
    points: np.ndarray,
    bounds: np.ndarray,
    resolution: float,
    shape: np.ndarray,
) -> np.ndarray:
    path_map = np.empty((points.shape[0], shape.size), dtype=np.int64)
    inv_resolution = 1.0 / resolution
    for i in range(points.shape[0]):
        for d in range(shape.size):
            value = int(round((points[i, d] - bounds[d, 0]) * inv_resolution - 0.5))
            if value < 0:
                value = 0
            elif value >= shape[d]:
                value = shape[d] - 1
            path_map[i, d] = value
    return path_map


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

        self._shape_array = np.asarray(self.shape, dtype=np.int64)
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

    def _type_map_flat(self) -> np.ndarray:
        return np.ravel(self.type_map.data)

    def _esdf_flat(self) -> np.ndarray:
        return np.ravel(self._esdf)

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

        point_world = _grid_map_to_world(np.asarray(point, dtype=np.float64), self.bounds, self.resolution)
        return tuple(float(x) for x in point_world)

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
        
        point_array = np.asarray(point, dtype=np.float64)
        if discrete:
            point_map = _grid_world_to_map_int(point_array, self.bounds, self.resolution, self._shape_array)
            return tuple(int(x) for x in point_map)
        else:
            point_map = _grid_world_to_map_float(point_array, self.bounds, self.resolution)
            return tuple(float(x) for x in point_map)

    def get_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """
        Get the distance between two points.

        Args:
            p1: Start point.
            p2: Goal point.
        
        Returns:
            dist: Distance between two points.
        """
        if len(p1) != len(p2):
            raise ValueError("Dimension mismatch")
        return _grid_distance(np.asarray(p1, dtype=np.float64), np.asarray(p2, dtype=np.float64))

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
        return _grid_within_bounds(np.asarray(point, dtype=np.int64), self._shape_array)

    def is_expandable(self, point: Tuple[int, ...], src_point: Tuple[int, ...] = None) -> bool:
        """
        Check if a point is expandable.
        
        Args:
            point: Point to check.
            src_point: Source point.
        
        Returns:
            expandable: True if the point is expandable, False otherwise.
        """
        point_array = np.asarray(point, dtype=np.int64)
        has_src_point = src_point is not None
        src_array = point_array if src_point is None else np.asarray(src_point, dtype=np.int64)

        return _grid_is_expandable(
            point_array,
            src_array,
            has_src_point,
            self._shape_array,
            self._type_map_flat(),
            self._esdf_flat(),
            TYPES.OBSTACLE,
            TYPES.INFLATION,
        )

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
        
        offsets = self._diagonal_offsets_array if diagonal else self._orthogonal_offsets_array
        positions, mask = _grid_neighbor_positions_and_mask(
            np.asarray(node.current, dtype=np.int64),
            offsets,
            self._shape_array,
            self._type_map_flat(),
            self._esdf_flat(),
            TYPES.OBSTACLE,
            TYPES.INFLATION,
        )

        return [
            Node(tuple(int(x) for x in positions[i]), node.current, node.g, node.h)
            for i in range(positions.shape[0])
            if mask[i]
        ]

    def line_of_sight(self, p1: Tuple[int, ...], p2: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """
        N-dimensional line of sight (Bresenham's line algorithm)
        
        Args:
            p1: Start point of the line.
            p2: End point of the line.
        
        Returns:
            points: List of point on the line of sight.
        """
        p1_array = np.asarray(p1, dtype=np.int64)
        p2_array = np.asarray(p2, dtype=np.int64)
        if p1_array.shape != p2_array.shape:
            p2_array - p1_array

        points = _grid_line_of_sight(p1_array, p2_array)
        return [tuple(int(x) for x in points[i]) for i in range(points.shape[0])]

    def in_collision(self, p1: Tuple[int, ...], p2: Tuple[int, ...]) -> bool:
        """
        Check if the line of sight between two points is in collision.
        
        Args:
            p1: Start point of the line.
            p2: End point of the line.
        
        Returns:
            in_collision: True if the line of sight is in collision, False otherwise.
        """
        p1_array = np.asarray(p1, dtype=np.int64)
        p2_array = np.asarray(p2, dtype=np.int64)
        if p1_array.shape != p2_array.shape:
            p2_array - p1_array

        return _grid_in_collision(
            p1_array,
            p2_array,
            self._shape_array,
            self._type_map_flat(),
            self._esdf_flat(),
            TYPES.OBSTACLE,
            TYPES.INFLATION,
        )

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
        path = list(path)
        if not path:
            return []

        points = np.asarray(path, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != self.dim:
            raise ValueError("Point dimension does not match map dimension.")

        path_world = _grid_path_map_to_world(points, self.bounds, self.resolution)
        return [tuple(float(x) for x in path_world[i]) for i in range(path_world.shape[0])]

    def path_world_to_map(self, path: List[Tuple[float, ...]], discrete: bool = True) -> List[tuple]:
        """
        Convert path from world coordinates to map coordinates

        Args:
            path: a list of world coordinates
            discrete: whether to round the coordinates to the nearest integer
        
        Returns:
            path: a list of map coordinates
        """
        path = list(path)
        if not path:
            return []

        points = np.asarray(path, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != self.dim:
            raise ValueError("Point dimension does not match map dimension.")

        if discrete:
            path_map = _grid_path_world_to_map_int(points, self.bounds, self.resolution, self._shape_array)
            return [tuple(int(x) for x in path_map[i]) for i in range(path_map.shape[0])]
        else:
            path_map = _grid_path_world_to_map_float(points, self.bounds, self.resolution)
            return [tuple(float(x) for x in path_map[i]) for i in range(path_map.shape[0])]

    def point_float_to_int(self, point: Tuple[float, ...]) -> Tuple[int, ...]:
        """
        Convert a point from float to integer coordinates.

        Args:
            point: a point in float coordinates
        
        Returns:
            point: a point in integer coordinates
        """
        point_int = _grid_point_float_to_int(np.asarray(point, dtype=np.float64), self._shape_array)
        return tuple(int(x) for x in point_int)

    def _precompute_offsets(self):
        # Generate all possible offsets (-1, 0, +1) in each dimension
        self._diagonal_offsets_array = np.array(np.meshgrid(*[[-1, 0, 1]]*self.dim), dtype=np.int64).T.reshape(-1, self.dim)
        # Remove the zero offset (current node itself)
        self._diagonal_offsets_array = self._diagonal_offsets_array[np.any(self._diagonal_offsets_array != 0, axis=1)]
        # self._diagonal_offsets = [Node((offset.tolist(), dtype=self.dtype)) for offset in self._diagonal_offsets]
        self._diagonal_offsets = [Node(tuple(offset.tolist())) for offset in self._diagonal_offsets_array]

        # Generate only orthogonal offsets (one dimension changes by ±1)
        self._orthogonal_offsets_array = np.zeros((2*self.dim, self.dim), dtype=np.int64)
        for d in range(self.dim):
            self._orthogonal_offsets_array[2*d, d] = 1
            self._orthogonal_offsets_array[2*d+1, d] = -1
        # self._orthogonal_offsets = [Node((offset.tolist(), dtype=self.dtype)) for offset in self._orthogonal_offsets]
        self._orthogonal_offsets = [Node(tuple(offset.tolist())) for offset in self._orthogonal_offsets_array]
