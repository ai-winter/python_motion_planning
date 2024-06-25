"""
@file: lqr.py
@breif: Linear Quadratic Regulator(LQR) motion planning
@author: Yang Haodong, Wu Maojia
@update: 2024.6.25
"""
import numpy as np

from .local_planner import LocalPlanner
from python_motion_planning.utils import Env

class LQR(LocalPlanner):
    """
    Class for Linear Quadratic Regulator(LQR) motion planning.

    Parameters:
        start (tuple): start point coordinate
        goal (tuple): goal point coordinate
        env (Env): environment
        heuristic_type (str): heuristic function type
        **params: other parameters can be found in the parent class LocalPlanner

    Examples:
        >>> from python_motion_planning.utils import Grid
        >>> from python_motion_planning.local_planner import LQR
        >>> start = (5, 5, 0)
        >>> goal = (45, 25, 0)
        >>> env = Grid(51, 31)
        >>> planner = LQR(start, goal, env)
        >>> planner.run()
    """
    def __init__(self, start: tuple, goal: tuple, env: Env, heuristic_type: str = "euclidean", **params) -> None:
        super().__init__(start, goal, env, heuristic_type, **params)
        # LQR parameters
        self.Q = np.diag([1, 1, 1])
        self.R = np.diag([1, 1])
        self.lqr_iteration = 100
        self.eps_iter = 1e-1

        # global planner
        g_start = (start[0], start[1])
        g_goal  = (goal[0], goal[1])
        self.g_planner = {"planner_name": "a_star", "start": g_start, "goal": g_goal, "env": env}
        self.path = self.g_path[::-1]

    def __str__(self) -> str:
        return "Linear Quadratic Regulator (LQR)"

    def plan(self):
        """
        LQR motion plan function.

        Returns:
            flag (bool): planning successful if true else failed
            pose_list (list): history poses of robot
        """
        dt = self.params["TIME_STEP"]
        for _ in range(self.params["MAX_ITERATION"]):
            # break until goal reached
            if self.reachGoal(tuple(self.robot.state.squeeze(axis=1)[0:3]), self.goal):
                return True, self.robot.history_pose

            # get the particular point on the path at the lookahead distance
            lookahead_pt, theta_trj, kappa = self.getLookaheadPoint()

            # calculate velocity command
            e_theta = self.regularizeAngle(self.robot.theta - self.goal[2])
            if not self.shouldMoveToGoal(self.robot.position, self.goal):
                if not self.shouldRotateToPath(abs(e_theta)):
                    u = np.array([[0], [0]])
                else:
                    u = np.array([[0], [self.angularRegularization(e_theta / dt)]])
            else:
                e_theta = self.regularizeAngle(
                    self.angle(self.robot.position, lookahead_pt) - self.robot.theta
                )
                if self.shouldRotateToPath(abs(e_theta)):
                    u = np.array([[0], [self.angularRegularization(e_theta / dt)]])
                else:
                    s = (self.robot.px, self.robot.py, self.robot.theta) # current state
                    s_d = (lookahead_pt[0], lookahead_pt[1], theta_trj)  # desired state
                    u_r = (self.robot.v, self.robot.v * kappa)           # refered input
                    u = self.lqrControl(s, s_d, u_r)

            # feed into robotic kinematic
            self.robot.kinematic(u, dt)
        
        return False, None

    def run(self):
        """
        Running both plannig and animation.
        """
        _, history_pose = self.plan()
        if not history_pose:
            raise ValueError("Path not found and planning failed!")

        path = np.array(history_pose)[:, 0:2]
        cost = np.sum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1, keepdims=True)))
        self.plot.plotPath(self.path, path_color="r", path_style="--")
        self.plot.animation(path, str(self), cost, history_pose=history_pose)

    def lqrControl(self, s: tuple, s_d: tuple, u_r: tuple) -> np.ndarray:
        """
        Execute LQR control process.

        Parameters:
            s (tuple): current state
            s_d (tuple): desired state
            u_r (tuple): refered control

        Returns:
            u (np.ndarray): control vector
        """
        dt = self.params["TIME_STEP"]

        # state equation on error
        A = np.identity(3)
        A[0, 2] = -u_r[0] * np.sin(s_d[2]) * dt
        A[1, 2] = u_r[0] * np.cos(s_d[2]) * dt

        B = np.zeros((3, 2))
        B[0, 0] = np.cos(s_d[2]) * dt
        B[1, 0] = np.sin(s_d[2]) * dt
        B[2, 1] = dt

        # discrete iteration Ricatti equation
        P, P_ = np.zeros((3, 3)), np.zeros((3, 3))
        P = self.Q

        # iteration
        for _ in range(self.lqr_iteration):
            P_ = self.Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(self.R + B.T @ P @ B) @ B.T @ P @ A
            if np.max(P - P_) < self.eps_iter:
                break
            P = P_

        # feedback
        K = -np.linalg.inv(self.R + B.T @ P_ @ B) @ B.T @ P_ @ A
        e = np.array([[s[0] - s_d[0]], [s[1] - s_d[1]], [self.regularizeAngle(s[2] - s_d[2])]])
        u = np.array([[u_r[0]], [u_r[1]]]) + K @ e

        return np.array([
            [self.linearRegularization(float(u[0]))], 
            [self.angularRegularization(float(u[1]))]
        ])