"""
@file: bspline_curve.py
@breif: B-Spline curve generation
@author: Winter
@update: 2023.7.29
"""
import math
import numpy as np

from typing import List

from .curve import Curve

from python_motion_planning.common.geometry import Point3d

class BSpline(Curve):
    """
    Class for B-Spline curve generation.

    Parameters:
        step (float): Simulation or interpolation size
        k (int): Degree of curve

    Examples:
        >>> from python_motion_planning.curve_generation import BSpline
        >>>	points = [(0, 0, 0), (10, 10, -90), (20, 5, 60)]
        >>> generator = BSpline(step, k)
        >>> generator.run(points)
    """
    def __init__(self, step: float, k: int, param_mode: str="centripetal", 
                 spline_mode: str="interpolation") -> None:
        super().__init__(step)
        self.k = k

        assert param_mode == "centripetal" or param_mode == "chord_length" \
               or param_mode == "uniform_spaced", "Parameter selection mode error!"
        self.param_mode = param_mode

        assert spline_mode == "interpolation" or spline_mode == "approximation", \
               "Spline mode selection error!"
        self.spline_mode = spline_mode
    
    def __str__(self) -> str:
        return "B-Spline Curve"
    
    def baseFunction(self, i: int, k: int, t: float, knot: List[float]) -> float:
        """
        Calculate base function using Cox-deBoor function.

        Parameters:
            i (int): The index of base function
            k (int): The degree of curve
            t (float): parameter
            knot (list[float]): knot vector

        Returns:
            Nik_t (float): The value of base function Nik(t)
        """
        Nik_t = 0
        if k == 0:
            Nik_t = 1.0 if t >= knot[i] and t < knot[i + 1] else 0.0
        else:
            length1 = knot[i + k] - knot[i]
            length2 = knot[i + k + 1] - knot[i + 1]
            if not length1 and not length2:
                Nik_t = 0
            elif not length1:
                Nik_t = (knot[i + k + 1] - t) / length2 * self.baseFunction(i + 1, k - 1, t, knot)
            elif not length2:
                Nik_t = (t - knot[i]) / length1 * self.baseFunction(i, k - 1, t, knot)
            else:
                Nik_t = (t - knot[i]) / length1 * self.baseFunction(i, k - 1, t, knot) + \
                        (knot[i + k + 1] - t) / length2 * self.baseFunction(i + 1, k - 1, t, knot)
        return Nik_t

    def paramSelection(self, points: List[Point3d]) -> List[float]:
        """
        Calculate parameters using the `uniform spaced` or `chrod length`
        or `centripetal` method.

        Parameters:
            points (list[Point3d]): path points

		Returns:
		    Parameters (list[float]): The parameters of given points
        """
        n = len(points)
        x_list = [pt.x() for pt in points]
        y_list = [pt.y() for pt in points]
        dx, dy = np.diff(x_list), np.diff(y_list)

        if self.param_mode == "uniform_spaced":
            return np.linspace(0, 1, n).tolist()

        elif self.param_mode == "chord_length":
            parameters = np.zeros(n)
            s = np.cumsum([math.hypot(idx, idy) for (idx, idy) in zip(dx, dy)])
            for i in range(1, n):
                parameters[i] = s[i - 1] / s[-1]
            return parameters.tolist()
        
        elif self.param_mode == "centripetal":
            alpha = 0.5
            s = np.cumsum([math.pow(math.hypot(idx, idy), alpha) for (idx, idy) in zip(dx, dy)])
            parameters = np.zeros(n)
            for i in range(1, n):
                parameters[i] = s[i - 1] / s[-1]
            return parameters.tolist()

    def knotGeneration(self, param: List[float], n: int) -> List[float]:
        """
        Generate knot vector.

        Parameters:
            param (list[float]): The parameters of given points
            n (int): The number of data points
		
		Returns:
		    knot (list[float]): The knot vector
        """
        m = n + self.k + 1
        knot = np.zeros(m)
        for i in range(self.k + 1):
            knot[i] = 0
        for i in range(n, m):
            knot[i] = 1
        for i in range(self.k + 1, n):
            for j in range(i - self.k, i):
                knot[i] = knot[i] + param[j]
            knot[i] = knot[i] / self.k
        return knot.tolist()
    
    def interpolation(self, points: List[Point3d], param: List[float], knot: List[float]) -> List[Point3d]:
        """
        Given a set of N data points, D0, D1, ..., Dn and a degree k,
        find a B-spline curve of degree k defined by N control points
        that passes all data points in the given order.

        Parameters:
            points (list[Point3d]): path points
            param (list[float]): The parameters of given points
            knot (list[float]): The knot vector
        
        Returns:
            control_points (list[Point3d]): The control points
        """
        n = len(points)
        N = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                N[i][j] = self.baseFunction(j, self.k, param[i], knot)
        N[n-1][n-1] = 1
        N_inv = np.linalg.inv(N)
        D = np.array([(pt.x(), pt.y()) for pt in points])
        control_pts = N_inv @ D
        return [Point3d(float(control_pts[i, 0]), float(control_pts[i, 1])) for i in range(n)]

    def approximation(self, points: List[Point3d], param: List[float], knot: List[float]) -> Point3d:
        """
        Given a set of N data points, D0, D1, ..., Dn, a degree k,
        and a number H, where N > H > k >= 1, find a B-spline curve
        of degree k defined by H control points that satisfies the
        following conditions:
            1. this curve contains the first and last data points;
            2. this curve approximates the data polygon in the sense
            of least square
        
        Parameters:
            points (list[Point3d]): path points
            param (list[float]): The parameters of given points
            knot (list[float]): The knot vector
        
		Returns:
		    control_points (list[Point3d]): The control points
        """
        n = len(points)
        D = np.array([(pt.x(), pt.y()) for pt in points])

        # heuristically setting the number of control points
        h = n - 1

        N = np.zeros((n, h))
        for i in range(n):
            for j in range(h):
                N[i][j] = self.baseFunction(j, self.k, param[i], knot)
        N_ = N[1 : n - 1, 1 : h - 1]

        qk = np.zeros((n - 2, 2))
        for i in range(1, n - 1):
            qk[i - 1] = D[i, :] - N[i][0] * D[0, :] - N[i][h - 1] * D[-1, :]
        Q = N_.T @ qk

        P = np.linalg.inv(N_.T @ N_) @ Q
        P = np.insert(P, 0, D[0, :], axis=0)
        P = np.insert(P, len(P), D[-1, :], axis=0)

        return [Point3d(float(P[i, 0]), float(P[i, 1])) for i in range(h)]

    def generation(self, t: List[float], k: int, knot: List[float], control_pts: List[Point3d]) -> List[Point3d]:
        """
        Generate the B-spline curve.

        Parameters:
            t (list[float]): The parameter values
            k (int): The degree of the B-spline curve
            knot (list[float]): The knot vector
            control_pts (list[Point3d]): The control points

        Returns:
            curve (list[Point3d]): The B-spline curve
        """
        N = np.zeros((len(t), len(control_pts)))

        for i in range(len(t)):
            for j in range(len(control_pts)):
                N[i][j] = self.baseFunction(j, k, t[i], knot)
        N[len(t) - 1][len(control_pts) - 1] = 1
        curve_pts = N @ np.array([(pt.x(), pt.y()) for pt in control_pts])

        return [Point3d(float(curve_pts[i, 0]), float(curve_pts[i, 1])) for i in range(len(t))]

    def run(self, points: List[Point3d]):
        """
        Running both generation and animation.

        Parameters:
            points (list[Point3d]): path points
        """
        assert len(points) >= 2, "Number of points should be at least 2."

        t = np.linspace(0, 1, int(1 / self.step)).tolist()
        params = self.paramSelection(points)
        knot = self.knotGeneration(params, len(points))

        if self.spline_mode == "interpolation":
            control_pts = self.interpolation(points, params, knot)
        elif self.spline_mode == "approximation":
            control_pts = self.approximation(points, params, knot)
            params = self.paramSelection(control_pts)
            knot = self.knotGeneration(params, len(control_pts))
        else:
            raise NotImplementedError
        
        control_x = [pt.x() for pt in control_pts]
        control_y = [pt.y() for pt in control_pts]

        path = self.generation(t, self.k, knot, control_pts)
        path_x = [pt.x() for pt in path]
        path_y = [pt.y() for pt in path]

        return [path_x, path_y], [
			{"type": "path", "name": "normal", "props": {"style": "-", "color": "#1f77b4"},
			 "data": [(ix, iy) for (ix, iy) in zip(path_x, path_y)],
			},
			{"type": "path", "name": "normal", "props": {"style": "--", "color": "#dddddd", "marker": "o"},
			 "data": [(ix, iy) for (ix, iy) in zip(control_x, control_y)],
			},
			{"type": "marker", "name": "normal", "props": {"marker": "x", "color": "#ff0000", "size": 40},
			 "data": [[pt.x() for pt in points], [pt.y() for pt in points]]
			}
		]