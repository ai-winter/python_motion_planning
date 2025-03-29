"""
@file: world.py
@breif: Motion Planning Base World
@author: Wu Maojia
@update: 2025.3.29
"""
from typing import Iterable
from abc import ABC, abstractmethod

import numpy as np


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


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)