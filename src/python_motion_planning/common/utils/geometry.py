"""
@file: a_star.py
@breif: A* planner
@author: Wu Maojia
@update: 2025.9.6
"""
import math

def dist(p1: tuple, p2: tuple, type: str = 'Euclidean') -> float:
    """
    Calculate the distance between two points

    Parameters:
        p1: First point
        p2: Second point
        type: Type of distance calculation, either 'Euclidean' or 'Manhattan'

    Returns:
        dist: Distance between the two points
    """
    if len(p1) != len(p2):
        raise ValueError("Dimension mismatch")
    if type == 'Euclidean':
        return math.sqrt(sum((a - b)** 2 for a, b in zip(p1, p2)))
    elif type == 'Manhattan':
        return sum(abs(a - b) for a, b in zip(p1, p2))
    else:
        raise ValueError("Invalid distance type")

def angle(v1: tuple, v2: tuple) -> float:
    """
    Calculate the angle between two vectors

    Parameters:
        v1: First vector
        v2: Second vector

    Returns:
        angle_rad: Angle in rad between the two vectors
    """
    if len(v1) != len(v2):
        raise ValueError("Dimension mismatch")
    
    dot_product = sum(a * b for a, b in zip(v1, v2))
    v1_norm = math.sqrt(sum(a**2 for a in v1))
    v2_norm = math.sqrt(sum(b**2 for b in v2))
    
    if  v1_norm == 0 or v2_norm == 0:
        raise ValueError("Zero vector cannot calculate angle")

    cos_theta = dot_product / (v1_norm * v2_norm)

    cos_theta = min(1.0, max(-1.0, cos_theta))

    angle_rad = math.acos(cos_theta)
    
    return angle_rad
