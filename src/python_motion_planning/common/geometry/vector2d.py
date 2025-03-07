"""
@file: vector2d.py
@breif: geometry: 2D vector.
@author: Winter
@update: 2025.1.7
"""
import math

from python_motion_planning.common.math import MathHelper

class Vec2d:
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

    def length(self) -> float:
        """
        Gets the length of the vector
        """
        return math.hypot(self.x_, self.y_)

    def lengthSqure(self) -> float:
        """
        Gets the squared length of the vector
        """
        return self.x_ ** 2 + self.y_ ** 2

    def angle(self) -> float:
        """
        Gets the angle between the vector and the positive x semi-axis
        """
        return math.atan2(self.y_, self.x_)

    def normalize(self) -> None:
        """
        Returns the unit vector that is co-linear with this vector
        """
        l = self.length()
        if l > MathHelper.kMathEpsilon:
            self.x_ /= l
            self.y_ /= l

    def distanceTo(self, other: 'Vec2d') -> float:
        """
        Returns the distance to the given vector
        """
        return math.hypot(self.x_ - other.x(), self.y_ - other.y())

    def distanceSqureTo(self, other: 'Vec2d') -> float:
        """
        Returns the squared distance to the given vector
        """
        dx = self.x_ - other.x()
        dy = self.y_ - other.y()
        return dx ** 2 + dy ** 2

    def crossProduct(self, other: 'Vec2d') -> float:
        """
        Returns the "cross" product between these two Vec2d (non-standard)
        """
        return self.x_ * other.y() - self.y_ * other.x()
    
    def innerProduct(self, other: 'Vec2d') -> float:
        """
        Returns the inner product between these two Vec2d
        """
        return self.x_ * other.x() + self.y_ * other.y()

    def rotate(self, angle: float) -> 'Vec2d':
        """
        Rotate the vector by angle
        """
        return Vec2d(
            self.x_ * math.cos(angle) - self.y_ * math.sin(angle),
            self.x_ * math.sin(angle) + self.y_ * math.cos(angle)
        )

    def selfRotate(self, angle: float) -> None:
        """
        Rotate the vector itself by angle
        """
        tmp_x = self.x_;
        self.x_ = self.x_ * math.cos(angle) - self.y_ * math.sin(angle)
        self.y_ = tmp_x * math.sin(angle) + self.y_ * math.cos(angle)

    def __add__(self, other: 'Vec2d') -> 'Vec2d':
        return Vec2d(self.x_ + other.x(), self.y_ + other.y())
    
    def __sub__(self, other: 'Vec2d') -> 'Vec2d':
        return Vec2d(self.x_ - other.x(), self.y_ - other.y())

    def __neg__(self) -> 'Vec2d':
        return Vec2d(-self.x_, -self.y_)

    def __mul__(self, ratio: float) -> 'Vec2d':
        return Vec2d(self.x_ * ratio, self.y_ * ratio)

    def __truediv__(self, ratio: float) -> 'Vec2d':
        if abs(ratio) > MathHelper.kMathEpsilon:
            return Vec2d(self.x_ / ratio, self.y_ / ratio)
        else:
            raise RuntimeError("Vector cannot divide by zero!")
    
    def __iadd__(self, other: 'Vec2d') -> 'Vec2d':
        self.x_ += other.x()
        self.y_ += other.y()
        return self
    
    def __isub__(self, other: 'Vec2d') -> 'Vec2d':
        self.x_ -= other.x()
        self.y_ -= other.y()
        return self

    def __imul__(self, ratio: float) -> 'Vec2d':
        self.x_ *= ratio
        self.y_ *= ratio
        return self

    def __rmul__(self, ratio: float) -> 'Vec2d':
        return self.__mul__(ratio)

    def __itruediv__(self, ratio: float) -> 'Vec2d':
        if abs(ratio) > MathHelper.kMathEpsilon:
            self.x_ /= ratio
            self.y_ /= ratio
            return self
        else:
            raise RuntimeError("Vector cannot divide by zero!")

    def __eq__(self, other: 'Vec2d') -> bool:
        return (abs(self.x_ - other.x()) < MathHelper.kMathEpsilon and \
                abs(self.y_ - other.y()) < MathHelper.kMathEpsilon)

    @staticmethod
    def createUnitVec2d(theta: float) -> 'Vec2d':
        return Vec2d(math.cos(theta), math.sin(theta))
