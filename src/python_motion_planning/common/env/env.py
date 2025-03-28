"""
@file: env.py
@breif: Motion Planning Base Environment
@author: Wu Maojia
@update: 2025.3.29
"""
from typing import Iterable
from abc import ABC, abstractmethod

class Env(ABC):
    """
    Class for Motion Planning Base Environment. It is continuous and in n-d Cartesian coordinate system.

    Parameters:
        size: size of environment (length of size means the number of dimensions)

    Examples:
        >>> env = Env((30, 40))

        >>> env
        Env((30, 40))

        >>> env.size
        (30, 40)

        >>> env.ndim
        2
    """
    def __init__(self, size: Iterable) -> None:
        super().__init__()
        try:
            self._size = tuple(size)
            self._ndim = len(self._size)

            if self._ndim <= 1:
                raise ValueError("Input length must be greater than 1.")
            if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in self._size):
                raise ValueError("Input must be a non-empty 1D array.")

        except Exception as e:
            raise ValueError("Invalid input for Env: {}".format(e))

    def __str__(self) -> str:
        return "Env({})".format(self._size)

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def size(self) -> tuple:
        return self._size

    @property
    def ndim(self) -> int:
        return self._ndim


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)