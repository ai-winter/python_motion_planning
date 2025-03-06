"""
@file: point2d.py
@breif: 2-dimension point data stucture
@author: Wu Maojia
@update: 2024.3.15
"""
import math


class Point2D(object):
    """
    Class for searching and manipulating 2-dimensional points.

    Parameters:
        x: x-coordinate of the 2d point
        y: y-coordinate of the 2d point
        eps: tolerance for float comparison

    Examples:
        >>> from python_motion_planning import Point2D
        >>> p1 = Point2D(1, 2)
        >>> p2 = Point2D(3, 4)
        ...
        >>> p1
        >>> Point2D(1, 2)
        ...
        >>> p1 + p2
        >>> Point2D(4, 6)
        ...
        >>> p1 - p2
        >>> Point2D(-2, -2)
        ...
        >>> p1 == p2
        >>> False
        ...
        >>> p1!= p2
        >>> True
        ...
        >>> p1.dist(p2)
        >>> 2.8284271247461903
        ...
        >>> p1.angle(p2)
        >>> 0.7853981633974483
    """

    def __init__(self, x: float, y: float, eps: float = 1e-6) -> None:
        self.x = x
        self.y = y
        self.eps = eps

        if abs(self.x - round(self.x)) < self.eps:
            self.x = round(self.x)

        if abs(self.y - round(self.y)) < self.eps:
            self.y = round(self.y)

    def __add__(self, point):
        assert isinstance(point, Point2D)
        return Point2D(self.x + point.x, self.y + point.y)

    def __sub__(self, point):
        assert isinstance(point, Point2D)
        return Point2D(self.x - point.x, self.y - point.y)

    def __eq__(self, point) -> bool:
        if not isinstance(point, Point2D):
            return False
        return abs(self.x - point.x) < self.eps and abs(self.y - point.y) < self.eps

    def __ne__(self, point) -> bool:
        return not self.__eq__(point)

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __str__(self) -> str:
        return "Point2D({}, {})".format(self.x, self.y)

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def from_tuple(point: tuple):
        return Point2D(point[0], point[1])

    @property
    def to_tuple(self) -> tuple:
        return int(self.x), int(self.y)

    def dist(self, point) -> float:
        assert isinstance(point, Point2D)
        return math.hypot(self.x - point.x, self.y - point.y)

    def angle(self, point) -> float:
        assert isinstance(point, Point2D)
        return math.atan2(point.y - self.y, point.x - self.x)