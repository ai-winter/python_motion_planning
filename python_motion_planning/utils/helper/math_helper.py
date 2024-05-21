"""
@file: math_helper.py
@breif: Contains common/commonly used math function
@author: Yang Haodong, Wu Maojia
@update: 2024.5.20
"""
import math 

class MathHelper:
    @staticmethod
    def circleSegmentIntersection(p1: tuple, p2: tuple, r: float) -> list:
        x1, x2 = p1[0], p2[0]
        y1, y2 = p1[1], p2[1]

        dx, dy = x2 - x1, y2 - y1
        dr2 = dx * dx + dy * dy
        D = x1 * y2 - x2 * y1

        # the first element is the point within segment
        d1 = x1 * x1 + y1 * y1
        d2 = x2 * x2 + y2 * y2
        dd = d2 - d1

        delta_2 = r * r * dr2 - D * D
        if delta_2 < 0:  # no intersection
            return []

        delta = math.sqrt(delta_2)
        if (delta == 0):
            return [(D * dy / dr2, -D * dx / dr2)]
        else:   # delta > 0
            return [
                ((D * dy + math.copysign(1.0, dd) * dx * delta) / dr2,
                (-D * dx + math.copysign(1.0, dd) * dy * delta) / dr2),
                ((D * dy - math.copysign(1.0, dd) * dx * delta) / dr2,
                (-D * dx - math.copysign(1.0, dd) * dy * delta) / dr2)
            ]

    @staticmethod
    def closestPointOnLine(a: tuple, b: tuple, p: tuple = (0.0, 0.0)) -> tuple:
        """
        Find the closest intersection point (foot of a perpendicular) between point p and the line ab.

        Parameters:
            a (tuple): point a of the line
            b (tuple): point b of the line
            p (tuple): point p to find the closest intersection point

        References:
            [1] method 2 of https://www.youtube.com/watch?v=TPDgB6136ZE
        """
        ap = (p[0] - a[0], p[1] - a[1])
        ab = (b[0] - a[0], b[1] - a[1])
        af_coef = (ap[0] * ab[0] + ap[1] * ab[1]) / (ab[0] ** 2 + ab[1] ** 2)
        af = (af_coef * ab[0], af_coef * ab[1])
        f = (a[0] + af[0], a[1] + af[1])
        return f
    
    @staticmethod
    def clamp(val: float, min_val: float, max_val: float) -> float:
        if val < min_val:
            val = min_val
        if val > max_val :
            val = max_val
        return val
