"""
@file: pose2d.py
@breif: 2-dimension pose data stucture
@author: Wu Maojia
@update: 2024.3.15
"""
import math


class Pose2D(object):
    """
    Class for searching and manipulating 2-dimensional poses.

    Parameters:
        x: x-coordinate of the 2d pose
        y: y-coordinate of the 2d pose
        theta: orientation of the 2d pose in radians
        eps: tolerance for float comparison

    Examples:
        >>> from python_motion_planning import Pose2D
        >>> p1 = Pose2D(1, 2)
        >>> p2 = Pose2D(3, 4, 1)
        ...
        >>> p1
        >>> Pose2D(1, 2, 0)
        ...
        >>> p2
        >>> Pose2D(3, 4, 1)
        ...
        >>> p1 + p2
        >>> Pose2D(4, 6, 1)
        ...
        >>> p1 - p2
        >>> Pose2D(-2, -2, -1)
        ...
        >>> p1 == p2
        >>> False
        ...
        >>> p1!= p2
        >>> True
    """

    def __init__(self, x: float, y: float, theta: float = 0, eps: float = 1e-6) -> None:
        self.x = x
        self.y = y
        self.theta = theta
        self.eps = eps

        if abs(self.x - round(self.x)) < self.eps:
            self.x = round(self.x)

        if abs(self.y - round(self.y)) < self.eps:
            self.y = round(self.y)

        if abs(self.theta - round(self.theta)) < self.eps:
            self.theta = round(self.theta)

    def __add__(self, pose):
        assert isinstance(pose, Pose2D)
        return Pose2D(self.x + pose.x, self.y + pose.y, self.theta + pose.theta)

    def __sub__(self, pose):
        assert isinstance(pose, Pose2D)
        return Pose2D(self.x - pose.x, self.y - pose.y, self.theta - pose.theta)

    def __eq__(self, pose) -> bool:
        if not isinstance(pose, Pose2D):
            return False
        return (abs(self.x - pose.x) < self.eps and abs(self.y - pose.y) < self.eps
                and abs(self.theta - pose.theta) < self.eps)

    def __ne__(self, pose) -> bool:
        return not self.__eq__(pose)

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.theta))

    def __str__(self) -> str:
        return "Pose2D({}, {}, {})".format(self.x, self.y, self.theta)

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def from_tuple(pose: tuple):
        return Pose2D(pose[0], pose[1], pose[2])

    @property
    def to_tuple(self) -> tuple:
        return self.x, self.y, self.theta