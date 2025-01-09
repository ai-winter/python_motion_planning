"""
@file: test_curve.py
@breif: curve generation application examples
@author: Winter
@update: 2024.4.23
"""
import math

from python_motion_planning.common.utils import Visualizer
from python_motion_planning.common.geometry import CurveFactory, Point3d

if __name__ == '__main__':
	'''
	Build environment
	'''
	points = [
		Point3d(0, 0, 0),
		Point3d(10, 10, -math.pi / 2),
		Point3d(20, 5, math.pi / 3),
		Point3d(30, 10, 2 * math.pi / 3),
		Point3d(35, -5, math.pi / 6),
		Point3d(25, -10, -2 * math.pi / 3),
		Point3d(15, -15, 5 * math.pi / 9),
		Point3d(0, -10, -math.pi / 2)
	]

	# points = [
	# 	Point3d(-3, 3, 2 * math.pi / 3),
	# 	Point3d(10, -7, math.pi / 6),
	# 	Point3d(10, 13, math.pi / 6),
	# 	Point3d(20, 5, -5 * math.pi / 36),
	# 	Point3d(35, 10, math.pi),
	# 	Point3d(32, -10, math.pi),
	# 	Point3d(5, -12, math.pi / 2)
	# ]

	'''
	Curve generator
	'''	  
	curve_factory = CurveFactory()
	# generator = curve_factory("dubins", step=0.1, max_curv=0.25)
	# generator = curve_factory("reeds_shepp", step=0.1, max_curv=0.25)
	# generator = curve_factory("bezier", step=0.1, offset=3.0)
	# generator = curve_factory("bspline", step=0.01, k=3)
	# generator = curve_factory("cubic_spline", step=0.1)
	generator = curve_factory("polynomial", step=2, max_acc=1.0, max_jerk=0.5)

	_, output = generator.run(points)

	'''
	Visualization
	'''
	visualizer = Visualizer("test_curve_generation")
	visualizer.setTitle(f"Curve generation: {str(generator)}")
	for info in output:
		if info["type"] == "path":
			visualizer.plotPath(
				info["data"], name=info["name"], props=info["props"] if "props" in info.keys() else {}
			)
		if info["type"] == "marker":
			visualizer.plotMarkers(
				info["data"], name=info["name"], props=info["props"] if "props" in info.keys() else {}
			)
	visualizer.show()