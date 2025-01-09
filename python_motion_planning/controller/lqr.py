"""
@file: lqr.py
@breif: Linear Quadratic Regulator(LQR) motion planning
@author: Winter
@update: 2024.1.12
"""
import numpy as np

from .controller import Controller

from python_motion_planning.common.utils import LOG
from python_motion_planning.common.geometry import Point3d
from python_motion_planning.common.structure import DiffRobot, DiffCmd

class LQR(Controller):
    """
    Class for Linear Quadratic Regulator(LQR) motion planning.

    Parameters:
        params (dict): parameters
    """
    def __init__(self, params: dict) -> None:
        super().__init__(params)
        # common
        self.start, self.goal, self.robot = None, None, None
        # LQR parameters
        self.Q = np.diag(self.params["Q_diag"])
        self.R = np.diag(self.params["R_diag"])
        self.lqr_iteration = self.params["lqr_iterations"]
        self.eps_iter = self.params["lqr_solve_eps"]

    def __str__(self) -> str:
        return "Linear Quadratic Regulator (LQR)"

    def plan(self, path: list):
        """
        LQR motion plan function.

        Parameters:
            path (list): planner path from path planning module

        Returns:
            flag (bool): planning successful if true else failed
            real_path (list): real tracking path
            cost (float): real path cost
            pose_list (list): history poses of robot
        """
        lookahead_pts = []
        self.start, self.goal = path[0], path[-1]
        self.robot = DiffRobot(self.start.x(), self.start.y(), self.start.theta(), 0, 0)
        dt = self.params["time_step"]
        for _ in range(self.params["max_iteration"]):
            # break until goal reached
            robot_pose = Point3d(self.robot.px, self.robot.py, self.robot.theta)
            if self.shouldRotateToGoal(robot_pose, self.goal):
                real_path = np.array(self.robot.history_pose)[:, 0:2]
                cost = np.sum(np.sqrt(np.sum(np.diff(real_path, axis=0)**2, axis=1, keepdims=True)))
                LOG.INFO(f"{str(self)} Controller Controlling Successfully!")
                return [
                    {"type": "value", "data": True, "name": "success"},
                    {"type": "value", "data": cost, "name": "cost"},
                    {"type": "path", "data": real_path, "name": "normal"},
                    {"type": "frame", "data": self.robot.history_pose, "name": "agent"},
                    {"type": "frame", "data": [[[pt.x()], [pt.y()]] for pt in lookahead_pts], "name": "marker"},
                    {"type": "frame", "name": "line", "props": {"color": "#f00"},
                     "data": [[[real_path[:i, 0], real_path[:i, 1]]] for i in range(real_path.shape[0])]},
                ]

            # get the particular point on the path at the lookahead distance
            lookahead_pt, kappa = self.getLookaheadPoint(path)

            # calculate velocity command
            e_theta = self.regularizeAngle(self.robot.theta - self.goal[2]) / 10
            if self.shouldRotateToGoal(robot_pose, self.goal):
                if not self.shouldRotateToPath(abs(e_theta)):
                    u = DiffCmd(0, 0)
                else:
                    u = DiffCmd(0, self.angularRegularization(e_theta / dt))
            else:
                e_theta = self.regularizeAngle(self.angle(robot_pose, lookahead_pt) - self.robot.theta)
                if self.shouldRotateToPath(abs(e_theta), np.pi / 4):
                    u = DiffCmd(0, self.angularRegularization(e_theta / dt / 10))
                else:
                    u_r = DiffCmd(self.robot.v, self.robot.v * kappa) # refered input
                    u = self.lqrControl(robot_pose, lookahead_pt, u_r)

            # update lookahead points
            lookahead_pts.append(lookahead_pt)

            # feed into robotic kinematic
            self.robot.kinematic(u, dt)
        
        LOG.WARN(f"{str(self)} Controller Controlling Failed!")
        return [
            {"type": "value", "data": False, "name": "success"},
            {"type": "value", "data": 0, "name": "cost"},
            {"type": "path", "data": [], "name": "normal"},
            {"type": "frame", "data": [], "name": "agent"},
            {"type": "frame", "data": [], "name": "marker"},
            {"type": "frame", "data": [], "name": "line"},
        ]

    def lqrControl(self, s: Point3d, s_d: Point3d, u_r: DiffCmd) -> np.ndarray:
        """
        Execute LQR control process.

        Parameters:
            s (Point3d): current state
            s_d (Point3d): desired state
            u_r (tuple): refered control

        Returns:
            u (np.ndarray): control vector
        """
        dt = self.params["time_step"]

        # state equation on error
        A = np.identity(3)
        A[0, 2] = -u_r.v * np.sin(s_d.theta()) * dt
        A[1, 2] = u_r.v * np.cos(s_d.theta()) * dt

        B = np.zeros((3, 2))
        B[0, 0] = np.cos(s_d.theta()) * dt
        B[1, 0] = np.sin(s_d.theta()) * dt
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
        e = np.array([[s.x() - s_d.x()], [s.y() - s_d.y()], [self.regularizeAngle(s.theta() - s_d.theta())]])
        u = np.array([[u_r.v], [u_r.w]]) + K @ e

        return DiffCmd(self.linearRegularization(float(u[0])), self.angularRegularization(float(u[1])))