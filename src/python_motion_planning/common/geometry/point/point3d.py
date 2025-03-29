"""
@file: point3d.py
@breif: 3-dimension point data stucture
@author: Wu Maojia
@update: 2025.3.28
"""
import math
from typing import Iterable, Union, Tuple

import numpy as np

from python_motion_planning.common.geometry.point import PointND


class Point3D(PointND):
    """
    Class for 3-dimensional points, inheriting from PointND.

    Parameters:
        x: x-coordinate or iterable of [x, y, z]
        y: y-coordinate (optional if x is iterable)
        z: z-coordinate (optional if x is iterable)
        dtype: numpy dtype (default: np.float64)

    Examples:
        >>> p1 = Point3D(1, 2, 3)
        >>> p2 = Point3D([4, 5, 6])
        
        >>> p1.x
        1.0
        
        >>> p1.y
        2.0
        
        >>> p1.z
        3.0
        
        >>> p1 + p2
        Point3D([5.0, 7.0, 9.0])
        
        >>> p1.cross(p2)
        Point3D([-3.0, 6.0, -3.0])

        >>> p1.dot(p2)
        32.0
    """
    def __init__(self, x: Union[float, Iterable], y: float = None, z: float = None, dtype: np.dtype = np.float64):
        if y is None and z is None:
            # Single argument case (x is iterable)
            super().__init__(x, dtype=dtype)
            if self.ndim != 3:
                raise ValueError("Point3D requires exactly 3 dimensions")
        else:
            # Three arguments case (x, y, z coordinates)
            if y is None or z is None:
                raise ValueError("Point3D requires all three coordinates")
            super().__init__([x, y, z], dtype=dtype)

    @property
    def x(self) -> float:
        """Get x coordinate"""
        return self._vec[0].item()

    @x.setter
    def x(self, value: float) -> None:
        """Set x coordinate"""
        self._vec[0] = value

    @property
    def y(self) -> float:
        """Get y coordinate"""
        return self._vec[1].item()

    @y.setter
    def y(self, value: float) -> None:
        """Set y coordinate"""
        self._vec[1] = value

    @property
    def z(self) -> float:
        """Get z coordinate"""
        return self._vec[2].item()

    @z.setter
    def z(self, value: float) -> None:
        """Set z coordinate"""
        self._vec[2] = value

    def cross(self, point: 'Point3D') -> 'Point3D':
        """
        Compute 3D cross product

        Parameters:
            point: Point3D instance to compute cross product with
        
        Returns:
            Point3D: cross product of two 3D vectors
        """
        if not isinstance(point, Point3D):
            raise TypeError("Operand must be a Point3D instance")
        return Point3D(np.cross(self._vec, point._vec), dtype=self.dtype)

    def __str__(self) -> str:
        return "Point3D([{}, {}, {}])".format(self.x, self.y, self.z)


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)