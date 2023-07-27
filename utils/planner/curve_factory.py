'''
@file: curve_factory.py
@breif: Facotry class for curve generation.
@author: Winter
@update: 2023.7.25
'''
import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))

from curve_generation import *

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
        else:
            raise ValueError("The `curve_name` must be set correctly.")