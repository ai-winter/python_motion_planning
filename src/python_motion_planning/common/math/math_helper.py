"""
@file: math_helper.py
@breif: math helper function.
@author: Winter
@update: 2025.1.7
"""
import math
import numpy as np

class MathHelper:
	kMathEpsilon = 1e-10

	def __init__(self) -> None:
		pass

	@staticmethod
	def isWithIn(val: float, bound1: float, bound2: float) -> bool:
		if bound1 > bound2:
			bound1, bound2 = bound2, bound1
		return val >= bound1 - MathHelper.kMathEpsilon and val <= bound2 + MathHelper.kMathEpsilon

	@staticmethod
	def crossProduct(start_point, end_point_1, end_point_2) -> float:
		return (end_point_1 - start_point).crossProduct(end_point_2 - start_point)
	
	@staticmethod
	def innerProduct(start_point, end_point_1, end_point_2) -> float:
		return (end_point_1 - start_point).innerProduct(end_point_2 - start_point)

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

	@staticmethod
	def less(v1: float, v2: float) -> bool:
		if v1 < v2 and abs(v1 - v2) > MathHelper.kMathEpsilon:
			return True
		else:
			return False

	@staticmethod
	def large(v1: float, v2: float) -> bool:
		if v1 > v2 and abs(v1 - v2) > MathHelper.kMathEpsilon:
			return True
		else:
			return False

	@staticmethod
	def equal(v1: float, v2: float) -> bool:
		if abs(v1 - v2) <= MathHelper.kMathEpsilon:
			return True
		else:
			return False

	@staticmethod
	def sortPoints(points: list) -> list:
		'''
		Sort a set of points in counterclockwise order
		'''
		sum_x, sum_y = 0.0, 0.0
		for p in points:
			sum_x += p.x()
			sum_y += p.y()

		middle_point = (sum_x / len(points), sum_y / len(points))
		pt_orien = list()
		for p in points:
			orien_v_x = p.x() - middle_point[0]
			orien_v_y = p.y() - middle_point[1]
			orien = np.arctan2(orien_v_y, orien_v_x)
			pt_orien.append((p, orien))
		pt_orien = sorted(pt_orien, key=lambda x: x[1])
		sorted_point = [x[0] for x in pt_orien]
		return sorted_point
