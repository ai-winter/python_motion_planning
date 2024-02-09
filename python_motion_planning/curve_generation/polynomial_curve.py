"""
@file: polynomial_curve.py
@breif: Polynomial curve generation
@author: Winter
@update: 2023.7.25
"""
import math
import numpy as np

from python_motion_planning.utils import Plot
from .curve import Curve

class Polynomial(Curve):
    """
    Class for polynomial curve generation(Quintic).

    Parameters:
        step (float): Simulation or interpolation size
        max_acc (float): Maximum acceleration
        max_jerk (float): Maximum jerk

    Examples:
        >>> from python_motion_planning.curve_generation import Polynomial
        >>>	points = [(0, 0, 0), (10, 10, -90), (20, 5, 60)]
        >>> generator = Polynomial(step, max_acc, max_jerk)
        >>> generator.run(points)
    """
    def __init__(self, step: float, max_acc: float, max_jerk: float) -> None:
        super().__init__(step)
        self.max_acc = max_acc
        self.max_jerk = max_jerk
        self.dt = 0.1
        self.t_min = 1
        self.t_max = 30
    
    def __str__(self) -> str:
        return "Quintic Polynomial Curve"
    
    class Poly:
        """
        Polynomial interpolation solver
        """
        def __init__(self, state0: tuple, state1: tuple, t: float) -> None:
            x0, v0, a0 = state0
            xt, vt, at = state1

            A = np.array([[t ** 3, t ** 4, t ** 5],
                          [3 * t ** 2, 4 * t ** 3, 5 * t ** 4],
                          [6 * t, 12 * t ** 2, 20 * t ** 3]])
            b = np.array([xt - x0 - v0 * t - a0 * t ** 2 / 2,
                          vt - v0 - a0 * t,
                          at - a0])
            X = np.linalg.solve(A, b)

            # Quintic polynomial coefficient
            self.p0 = x0
            self.p1 = v0
            self.p2 = a0 / 2.0
            self.p3 = X[0]
            self.p4 = X[1]
            self.p5 = X[2]

        def x(self, t):
            return self.p0 + self.p1 * t + self.p2 * t ** 2 + \
                self.p3 * t ** 3 + self.p4 * t ** 4 + self.p5 * t ** 5

        def dx(self, t):
            return self.p1 + 2 * self.p2 * t + 3 * self.p3 * t ** 2 + \
                4 * self.p4 * t ** 3 + 5 * self.p5 * t ** 4

        def ddx(self, t):
            return 2 * self.p2 + 6 * self.p3 * t + 12 * self.p4 * t ** 2 + 20 * self.p5 * t ** 3

        def dddx(self, t):
            return 6 * self.p3 + 24 * self.p4 * t + 60 * self.p5 * t ** 2
    
    class Trajectory:
        """
        Polynomial interpolation solver
        """
        def __init__(self):
            self.clear()
        
        def clear(self):
            self.time = []
            self.x = []
            self.y = []
            self.yaw = []
            self.v = []
            self.a = []
            self.jerk = []
        
        @property
        def size(self):
            assert len(self.time) == len(self.x) and \
                   len(self.x) == len(self.y) and    \
                   len(self.y) == len(self.yaw) and  \
                   len(self.yaw) == len(self.v) and  \
                   len(self.v) == len(self.a) and    \
                   len(self.a) == len(self.jerk),    \
                   "Unequal dimensions of each attribute, this should not happen."
            return len(self.time)

    def generation(self, start_pose: tuple, goal_pose: tuple):
        """
        Generate the polynomial Curve.

        Parameters:
            start_pose (tuple): Initial pose (x, y, yaw)
            goal_pose (tuple): Target pose (x, y, yaw)

        Returns:
            traj (Traj): The first trajectory that satisfies the acceleration and jerk constraint
        """
        sx, sy, syaw, sv, sa = start_pose
        gx, gy, gyaw, gv, ga = goal_pose
        
        sv_x = sv * math.cos(syaw)
        sv_y = sv * math.sin(syaw)
        gv_x = gv * math.cos(gyaw)
        gv_y = gv * math.sin(gyaw)

        sa_x = sa * math.cos(syaw)
        sa_y = sa * math.sin(syaw)
        ga_x = ga * math.cos(gyaw)
        ga_y = ga * math.sin(gyaw)
        
        traj = self.Trajectory()

        for T in np.arange(self.t_min, self.t_max, self.step):
            x_psolver = self.Poly((sx, sv_x, sa_x), (gx, gv_x, ga_x), T)
            y_psolver = self.Poly((sy, sv_y, sa_y), (gy, gv_y, ga_y), T)

            for t in np.arange(0.0, T + self.dt, self.dt):
                traj.time.append(t)
                traj.x.append(x_psolver.x(t))
                traj.y.append(y_psolver.x(t))

                vx = x_psolver.dx(t)
                vy = y_psolver.dx(t)
                traj.v.append(math.hypot(vx, vy))
                traj.yaw.append(math.atan2(vy, vx))

                ax = x_psolver.ddx(t)
                ay = y_psolver.ddx(t)
                a = math.hypot(ax, ay)
                if len(traj.v) >= 2 and traj.v[-1] - traj.v[-2] < 0.0:
                    a *= -1
                traj.a.append(a)

                jx = x_psolver.dddx(t)
                jy = y_psolver.dddx(t)
                j = math.hypot(jx, jy)
                if len(traj.a) >= 2 and traj.a[-1] - traj.a[-2] < 0.0:
                    j *= -1
                traj.jerk.append(j)

            if max(np.abs(traj.a)) <= self.max_acc and \
               max(np.abs(traj.jerk)) <= self.max_jerk:
                return traj
            else:
                traj.clear()

        return traj

    def run(self, points: list):
        """
        Running both generation and animation.

        Parameters:
            points (list[tuple]): path points
        """
        assert len(points) >= 2, "Number of points should be at least 2."
        import matplotlib.pyplot as plt

        # generate velocity and acceleration constraints heuristically
        v = [0]
        for i in range(len(points) - 1):
            v.append(1.0)

        a = [(v[i + 1] - v[i]) / 5 for i in range(len(points) - 1)]
        a.append(0)

        # generate curve
        path_x, path_y, path_yaw = [], [], []
        for i in range(len(points) - 1):
            path = self.generation(
				(points[i][0], points[i][1], np.deg2rad(points[i][2]), v[i], a[i]),
				(points[i + 1][0], points[i + 1][1], np.deg2rad(points[i + 1][2]), v[i + 1], a[i + 1])
            )

            for j in range(path.size):
                path_x.append(path.x[j])
                path_y.append(path.y[j])
                path_yaw.append(path.yaw[j])
        
        # animation
        plt.figure("curve generation")

        # # static
        # plt.plot(path_x, path_y, linewidth=2, c="#1f77b4")
        # for x, y, theta in points:
        #     Plot.plotArrow(x, y, np.deg2rad(theta), 2, 'blueviolet')

        # plt.axis("equal")
        # plt.title(str(self))

        # dynamic
        plt.ion()
        for i in range(len(path_x)):
            plt.clf()
            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(path_x, path_y, linewidth=2, c="#1f77b4")
            for x, y, theta in points:
                Plot.plotArrow(x, y, np.deg2rad(theta), 2, 'blueviolet')
            Plot.plotCar(path_x[i], path_y[i], path_yaw[i], 1.5, 3, "black")
            plt.axis("equal")
            plt.title(str(self))
            plt.draw()
            plt.pause(0.001)

        plt.show()



