"""
@file: pointnd.py
@breif: n-dimension point data stucture
@author: Wu Maojia
@update: 2025.3.28
"""
import numpy as np
from typing import Iterable, Union
import copy


class PointND(object):
    """
    Class for n-dimensional points.

    Parameters:
        vec: vector of point coordinates

    Examples:
        >>> p1 = PointND([1, 2], dtype=np.float64)
        >>> p2 = PointND([3, 4], dtype=np.float64)

        >>> p1
        PointND([1.0, 2.0], dtype=float64)

        >>> p1[0]
        1.0

        >>> p1 + p2
        PointND([4.0, 6.0], dtype=float64)

        >>> p1 - p2
        PointND([-2.0, -2.0], dtype=float64)

        >>> p1 == p2
        False

        >>> p1!= p2
        True

        >>> p1*3
        PointND([3.0, 6.0], dtype=float64)

        >>> p1.dot(p2)
        11.0

        >>> tuple(p1)
        (1.0, 2.0)

        >>> p1.dtype
        dtype('float64')

        >>> p1.ndim
        2

        >>> p1.astype(np.int32)
        PointND([1, 2], dtype=int32)

        >>> p1.dist(p2)
        2.8284271247461903

        >>> p1.dist(p2, type='Manhattan')
        4.0

        >>> p1.angle(p2)
        0.17985349979247847
    """

    def __init__(self, vec: Iterable, dtype: np.dtype = np.float64) -> None:
        try:
            self._vec = np.array(vec, dtype=dtype)
            if len(self._vec.shape) != 1 or self._vec.size == 0:
                raise ValueError("Input must be a non-empty 1D array")
        except Exception as e:
            raise ValueError("Invalid input for PointND: {}".format(e))

    def __add__(self, point: 'PointND') -> 'PointND':
        if not isinstance(point, PointND):
            raise TypeError("Operands must be PointND instances")
        if len(self._vec) != len(point._vec):
            raise ValueError("Dimension mismatch")
        return self.__class__(self._vec + point._vec)

    def __sub__(self, point: 'PointND') -> 'PointND':
        if not isinstance(point, PointND):
            raise TypeError("Operands must be PointND instances")
        if len(self._vec) != len(point._vec):
            raise ValueError("Dimension mismatch")
        return self.__class__(self._vec - point._vec)
    
    def __mul__(self, scalar: Union[int, float]) -> 'PointND':
        if not isinstance(scalar, (int, float)):
            raise TypeError("Can only multiply by scalar")
        return self.__class__(self._vec * scalar)

    def __eq__(self, point: 'PointND') -> bool:
        if not isinstance(point, PointND):
            return False
        return (self._vec.dtype == point._vec.dtype and 
                np.array_equal(self._vec, point._vec))

    def __ne__(self, point: 'PointND') -> bool:
        return not self.__eq__(point)

    def __hash__(self) -> int:
        return hash(tuple(self._vec))

    def __str__(self) -> str:
        return "PointND({}, dtype={})".format(self._vec.tolist(), self._vec.dtype)

    def __repr__(self) -> str:
        return self.__str__()

    def __iter__(self):
        return iter(self._vec.tolist())

    def __getitem__(self, idx):
        return self._vec[idx].item()

    def __copy__(self) -> 'PointND':
        return self.__class__(self._vec.copy(), self._vec.dtype)

    def __deepcopy__(self, memo: dict) -> 'PointND':
        return self.__class__(copy.deepcopy(self._vec), self._vec.dtype)

    @property
    def dtype(self) -> np.dtype:
        return self._vec.dtype

    @property
    def ndim(self) -> int:
        return len(self._vec)

    def dist(self, point: 'PointND', type: str = 'Euclidean') -> float:
        """
        Calculate the distance between two points

        Parameters:
            point: PointND instance to calculate the distance to
            type: Type of distance calculation, either 'Euclidean' or 'Manhattan'

        Returns:
            Distance between the two points
        """
        if not isinstance(point, PointND):
            raise TypeError("Input must be a PointND instance")
        if len(self._vec) != len(point._vec):
            raise ValueError("Dimension mismatch")
        if type == 'Euclidean':
            return np.linalg.norm(self._vec - point._vec).item()
        elif type == 'Manhattan':
            return np.sum(np.abs(self._vec - point._vec)).item()
        else:
            raise ValueError("Invalid distance type")

    def angle(self, point: 'PointND') -> float:
        """
        Calculate the angle between two points

        Parameters:
            point: PointND instance to calculate the angle to

        Returns:
            Angle between the two points in radians
        """
        if not isinstance(point, PointND):
            raise TypeError("Input must be a PointND instance")
        if len(self._vec) != len(point._vec):
            raise ValueError("Dimension mismatch")

        vec1 = self._vec.astype(np.float64)
        vec2 = point._vec.astype(np.float64)
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            raise ValueError("Cannot compute angle for zero vector")
            
        cos_theta = np.dot(vec1, vec2) / (norm1 * norm2)
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0)).item()

        return theta

    def dot(self, point: 'PointND') -> float:
        """
        Compute dot product

        Parameters:
            point: PointnD instance to compute dot product with
        
        Returns:
            float: dot product of two ND vectors
        """
        if not isinstance(point, PointND):
            raise TypeError("Operand must be a PointND instance")
        if len(self._vec) == 2 and len(point._vec) == 2:
            return (self._vec[0] * point._vec[0] + self._vec[1] * point._vec[1]).item()
        return float(np.dot(self._vec, point._vec))

    def astype(self, dtype: np.dtype) -> 'PointND':
        """
        Convert the PointND instance to a new instance with a different dtype
        
        Parameters:
            dtype: Data type to convert to
            
        Returns:
            New PointND instance with the specified dtype
        """
        point = self.__class__(self._vec.astype(dtype), dtype)
        return point


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)