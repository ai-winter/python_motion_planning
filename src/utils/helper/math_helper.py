'''
@file: math_helper.py
@breif: Contains common/commonly used math function
@author: Winter
@update: 2023.1.26
'''
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

        delta = math.sqrt(r * r * dr2 - D * D)

        if delta >= 0:
            if (delta == 0):
                return [(D * dy / dr2, -D * dx / dr2)]
            else:
                return [
                    ((D * dy + math.copysign(1.0, dd) * dx * delta) / dr2,
                    (-D * dx + math.copysign(1.0, dd) * dy * delta) / dr2),
                    ((D * dy - math.copysign(1.0, dd) * dx * delta) / dr2,
                    (-D * dx - math.copysign(1.0, dd) * dy * delta) / dr2)
                ]
        else:
            return []
    
    @staticmethod
    def clamp(val: float, min_val: float, max_val: float) -> float:
        if val < min_val:
            val = min_val
        if val > max_val :
            val = max_val
        return val
