"""
@file: point.py
@breif: geometry: point.
@author: Winter
@update: 2025.1.7
"""
from python_motion_planning.common.math import MathHelper

class Point2d:
    def __init__(self, x: float=0.0, y: float=0.0) -> None:
        self.x_ = x
        self.y_ = y
    
    """
    Getter and setter for x or y component
    """
    def x(self) -> float:                return self.x_
    def y(self) -> float:                return self.y_
    def setX(self, x: float) -> None:    self.x_ = x
    def setY(self, y: float) -> None:    self.y_ = y

    def __iter__(self):
        return iter((self.x_, self.y_))

    def __getitem__(self, index: int) -> float:
        val = self.x_ if index == 0  else self.y_
        return val
    
    def __eq__(self, other: 'Point2d') -> bool:
        return (abs(self.x_ - other.x()) < MathHelper.kMathEpsilon and \
            abs(self.y_ - other.y()) < MathHelper.kMathEpsilon)

    def __hash__(self) -> int:
        return hash((self.x_, self.y_))

class Point3d:
    def __init__(self, x: float=0.0, y: float=0.0, theta: float=0.0) -> None:
        self.x_ = x
        self.y_ = y
        self.theta_ = theta
    
    """
    Getter and setter for x, y or theta component
    """
    def x(self) -> float:                       return self.x_
    def y(self) -> float:                       return self.y_
    def theta(self) -> float:                   return self.theta_
    def setX(self, x: float) -> None:           self.x_ = x
    def setY(self, y: float) -> None:           self.y_ = y
    def setTheta(self, theta: float) -> None:   self.theta_ = theta

    def __iter__(self):
        return iter((self.x_, self.y_, self.theta_))

    def __getitem__(self, index: int) -> float:
        if index == 0:
            return self.x_
        elif index == 1:
            return self.y_
        else:
            return self.theta_
    
    def __eq__(self, other: 'Point3d') -> bool:
        if not isinstance(other, Point3d):
            return False
        return (abs(self.x_ - other.x()) < MathHelper.kMathEpsilon and \
            abs(self.y_ - other.y()) < MathHelper.kMathEpsilon and \
            abs(self.theta_ - other.theta()) < MathHelper.kMathEpsilon)
    
    def __hash__(self) -> int:
        return hash((self.x_, self.y_, self.theta_))