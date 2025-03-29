"""
@file: point2d.py
@breif: 2-dimension point data stucture
@author: Wu Maojia
@update: 2025.3.28
"""
import math
from typing import Iterable, Union, Tuple

import numpy as np

from python_motion_planning.common.geometry.point import PointND


class Point2D(PointND):
    """
    Class for 2-dimensional points, inheriting from PointND.

    Parameters:
        x: x-coordinate or iterable of [x, y]
        y: y-coordinate (optional if x is iterable)
        dtype: numpy dtype (default: np.float64)

    Examples:
        >>> p1 = Point2D(1, 2)
        >>> p2 = Point2D([3, 4])
        
        >>> p1.x
        1.0
        
        >>> p1.y
        2.0
        
        >>> p1 + p2
        Point2D([4.0, 6.0])
        
        >>> p1.cross(p2)
        -2.0

        >>> p1.dot(p2)
        11.0
    """
    def __init__(self, x: Union[float, Iterable], y: float = None, dtype: np.dtype = np.float64):
        if y is None:
            # Single argument case (x is iterable)
            super().__init__(x, dtype=dtype)
            if self.ndim != 2:
                raise ValueError("Point2D requires exactly 2 dimensions")
        else:
            # Two arguments case (x and y coordinates)
            super().__init__([x, y], dtype=dtype)

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

    def cross(self, point: 'Point2D') -> float:
        """
        Compute 2D cross product (also known as perp dot product)

        Parameters:
            point: Point2D instance to compute cross product with
        
        Returns:
            cross_product: cross product of two 2D vectors
        """
        if not isinstance(point, Point2D):
            raise TypeError("Operand must be a Point2D instance")
        return float(np.cross(self._vec, point._vec))

    def __str__(self) -> str:
        return "Point2D([{}, {}])".format(self.x, self.y)


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)