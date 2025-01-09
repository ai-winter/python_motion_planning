"""
@file: line_segment2d.py
@breif: 2D Line segment class.
@author: Winter
@update: 2025.1.7
"""
import math

from python_motion_planning.common.math import MathHelper

from .vector2d import Vec2d

class LineSegment2d:
    def __init__(self, start: Vec2d, end: Vec2d) -> None:
        '''
        Constructor with start point and end point.
        '''
        self.start_ = start
        self.end_ = end
        dx = end.x() - start.x()
        dy = end.y() - start.y()
        self.length_ = math.hypot(dx, dy)
        self.unit_dir_ = Vec2d(0, 0) if self.length_ < MathHelper.kMathEpsilon \
            else Vec2d(dx / self.length_, dy / self.length_)
        self.heading_ = self.unit_dir_.angle()
    
    """
    Getter for start or end component
    """
    def start(self) -> Vec2d:           return self.start_
    def end(self) -> Vec2d:             return self.end_
    def unit_direction(self) -> Vec2d:  return self.unit_dir_
    def center(self) -> Vec2d:          return 0.5 * (self.start_ + self.end_)
    def heading(self) -> float:         return self.heading_
    def cos_heading(self) -> float:     return self.unit_dir_.x()
    def sin_heading(self) -> float:     return self.unit_dir_.y()
    def length(self) -> float:          return self.length_
    def lengthSqure(self) -> float:     return self.length_ * self.length_

    def rotate(self, angle: float) -> Vec2d:
        '''
        Get a new line-segment with the same start point, but rotated counterclock-wise by the given amount.
        '''
        diff_vec = self.end_ - self.start_
        diff_vec.selfRotate(angle)
        return self.start_ + diff_vec

    def distanceTo(self, point: Vec2d, nearest_pt: Vec2d=None) -> float:
        """
        Compute the shortest distance from a point on the line segment
        to a point in 2-D, and get the nearest point on the line segment.
        """
        if self.length_ < MathHelper.kMathEpsilon:
            return point.distanceTo(self.start_)
        
        x0 = point.x() - self.start_.x()
        y0 = point.y() - self.start_.y()
        proj = x0 * self.unit_dir_.x() + y0 * self.unit_dir_.y()
        if proj <= 0.0:
            if nearest_pt is not None:
                nearest_pt = self.start_
            return math.hypot(x0, y0)
        if proj >= self.length_:
            if nearest_pt is not None:
                nearest_pt = self.end_
            return point.distanceTo(self.end_)

        if nearest_pt is not None:
            nearest_pt = self.start_ + self.unit_dir_ * proj

        return abs(x0 * self.unit_dir_.y() - y0 * self.unit_dir_.x())

    def distanceSquareTo(self, point: Vec2d, nearest_pt: Vec2d=None) -> float:
        """
        Compute the square of the shortest distance from a point on the line segment
        to a point in 2-D, and get the nearest point on the line segment.
        """
        if self.length_ < MathHelper.kMathEpsilon:
            return point.distanceTo(self.start_)
        
        x0 = point.x() - self.start_.x()
        y0 = point.y() - self.start_.y()
        proj = x0 * self.unit_dir_.x() + y0 * self.unit_dir_.y()
        if proj <= 0.0:
            if nearest_pt is not None:
                nearest_pt = self.start_
            return x0 ** 2 + y0 ** 2
        if proj >= self.length_:
            if nearest_pt is not None:
                nearest_pt = self.end_
            return point.distanceSqureTo(self.end_)

        if nearest_pt is not None:
            nearest_pt = self.start_ + self.unit_dir_ * proj
            
        return math.pow(x0 * self.unit_dir_.y() - y0 * self.unit_dir_.x(), 2)

    def isPointIn(self, point: Vec2d) -> bool:
        """
        Check if a point is within the line segment
        """
        if self.length_ <= MathHelper.kMathEpsilon:
            return abs(point.x() - self.start_.x()) <= MathHelper.kMathEpsilon and \
                abs(point.y() - self.start_.y()) <= MathHelper.kMathEpsilon
        prod = MathHelper.crossProduct(point, self.start_, self.end_)
        if abs(prod) > MathHelper.kMathEpsilon:
            return False
        return MathHelper.isWithIn(point.x(), self.start_.x(), self.end_.x()) and \
            MathHelper.isWithIn(point.y(), self.start_.y(), self.end_.y())

    def hasIntersect(self, other_segment: 'LineSegment2d') -> bool:
        """
        Check if the line segment has an intersect with another line segment in 2-D.
        """
        point = Vec2d()
        return self.getIntersect(other_segment, point)

    def getIntersect(self, other_segment: 'LineSegment2d', point: Vec2d) -> bool:
        """
        Compute the intersect with another line segment in 2-D if any.
        """
        if self.isPointIn(other_segment.start()):
            point = other_segment.start()
            return True
        if self.isPointIn(other_segment.end()):
            point = other_segment.end()
            return True
        if other_segment.isPointIn(self.start_):
            point = self.start_
            return True
        if other_segment.isPointIn(self.end_):
            point = self.end_
            return True
        if self.length_ < MathHelper.kMathEpsilon and other_segment.length() < MathHelper.kMathEpsilon:
            return False
        
        cc1 = MathHelper.crossProduct(self.start_, self.end_, other_segment.start())
        cc2 = MathHelper.crossProduct(self.start_, self.end_, other_segment.end())
        if cc1 * cc2 >= -MathHelper.kMathEpsilon:
            return False
        
        cc3 = MathHelper.crossProduct(other_segment.start(), other_segment.end(), self.start_)
        cc4 = MathHelper.crossProduct(other_segment.start(), other_segment.end(), self.end_)
        if cc3 * cc4 >= -MathHelper.kMathEpsilon:
            return False

        ratio = cc4 / (cc4 - cc3)
        point = Vec2d(
            self.start_.x() * ratio + self.end_.x() * (1.0 - ratio),
            self.start_.y() * ratio + self.end_.y() * (1.0 - ratio)
        )
        return True

    def projectOntoUnit(self, point: Vec2d) -> float:
        """
        Compute the projection of a vector onto the line segment
        """
        return self.unit_dir_.innerProduct(point - self.start_)

    def productOntoUnit(self, point: Vec2d) -> float:
        """
        Compute the cross product of a vector onto the line segment
        """
        return self.unit_dir_.crossProduct(point - self.start_)

    def getPerpendicularFoot(self, point: Vec2d, foot_point: Vec2d) -> float:
        """
        Compute perpendicular foot of a point in 2-D on the straight line expanded from the line segment.
        """
        if self.length_ <= MathHelper.kMathEpsilon:
            foot_point = self.start_
            return point.distanceTo(self.start_)
        x0 = point.x() - self.start_.x()
        y0 = point.y() - self.start_.y()
        proj = x0 * self.unit_direction_.x() + y0 * self.unit_direction_.y()
        foot_point = self.start_ + self.unit_direction_ * proj
        return abs(x0 * self.unit_direction_.y() - y0 * self.unit_direction_.x())