"""
@file: curve_factory.py
@breif: Facotry class for curve generation.
@author: Winter
@update: 2023.7.25
"""
from python_motion_planning.curve_generation import *

class CurveFactory(object):
    def __init__(self) -> None:
        pass

    def __call__(self, curve_name, **config):
        if curve_name == "dubins":
            return Dubins(**config)
        elif curve_name == "bezier":
            return Bezier(**config)
        elif curve_name == "polynomial":
            return Polynomial(**config)
        elif curve_name == "reeds_shepp":
            return ReedsShepp(**config)
        elif curve_name == "cubic_spline":
            return CubicSpline(**config)
        elif curve_name == "bspline":
            return BSpline(**config)
        elif curve_name == "fem_pos_smoother":
            return FemPosSmoother(**config)
        else:
            raise ValueError("The `curve_name` must be set correctly.")