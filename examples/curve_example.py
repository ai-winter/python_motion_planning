"""
@file: curve_example.py
@breif: curve generation application examples
@author: Winter
@update: 2023.7.25
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, "../")))
from python_motion_planning.utils import CurveFactory

if __name__ == '__main__':
	# simulation pose
	# points = [(0, 0, 0), (10, 10, -90), (20, 5, 60), (30, 10, 120),
	# 		(35, -5, 30), (25, -10, -120), (15, -15, 100), (0, -10, -90)]

	# points = [(-3, 3, 120), (10, -7, 30), (10, 13, 30), (20, 5, -25),
	# 		  (35, 10, 180), (32, -10, 180), (5, -12, 90)]

	points = [(0.5, 0.1), (1.0, 0.3), (2.0, 0.2), (3.0, 0.4), (4.0, 0.3),
			  (5.0, -0.2), (6.0, -0.1), (7.0, 0.0), (8.0, 0.5), (9.0, 0),
			  (10.0, 0.1), (11.0, 0.3), (12.0, 0.2), (13.0, 0.4), (14.0, 0.3),
			  (15.0, -0.2), (16.0, -0.1), (17.0, 0), (18.0, 0.5), (19.0, 0)
			  ]
			  
	# curve generation constructor
	curve_factory = CurveFactory()

	# create generator
	# generator = curve_factory("dubins", step=0.1, max_curv=0.25)
	# generator = curve_factory("bezier", step=0.1, offset=3.0)
	# generator = curve_factory("polynomial", step=2, max_acc=1.0, max_jerk=0.5)
	# generator = curve_factory("reeds_shepp", step=0.1, max_curv=0.25)
	# generator = curve_factory("cubic_spline", step=0.1)
	# generator = curve_factory("bspline", step=0.01, k=3)
	generator = curve_factory("fem_pos_smoother", w_smooth=10, w_ref=1, w_length=1, dx_l=0.2, dx_u=0.2, dy_l=0.2, dy_u=0.2)

	# animation
	generator.run(points)
	

