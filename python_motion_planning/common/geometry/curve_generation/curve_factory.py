"""
@file: curve_factory.py
@breif: Facotry class for curve generation.
@author: Winter
@update: 2023.7.25
"""
from .polynomial_curve import Polynomial
from .bezier_curve import Bezier
from .bspline_curve import BSpline
from .dubins_curve import Dubins
from .reeds_shepp import ReedsShepp
from .cubic_spline import CubicSpline

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
        else:
            raise ValueError("The `curve_name` must be set correctly.")