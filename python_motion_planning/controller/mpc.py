"""
@file: mpc.py
@breif: Model Predicted Control (MPC) motion planning
@author: Winter
@update: 2024.1.30
"""
import osqp
import numpy as np
from scipy import sparse

from .controller import Controller

from python_motion_planning.common.utils import LOG
from python_motion_planning.common.geometry import Point3d
from python_motion_planning.common.structure import DiffRobot, DiffCmd

class MPC(Controller):
    """
    Class for Model Predicted Control (MPC) motion planning.

    Parameters:
        params (dict): parameters
    """
    def __init__(self, params: dict) -> None:
        super().__init__(params)
        # common
        self.start, self.goal, self.robot = None, None, None
        # MPC parameters
        self.p = self.params["p"]
        self.m = self.params["m"]
        self.Q = np.diag(self.params["Q_diag"])
        self.R = np.diag(self.params["R_diag"])
        self.u_min = np.array([[self.params["min_v"]], [self.params["min_w"]]])
        self.u_max = np.array([[self.params["max_v"]], [self.params["max_w"]]])
        self.du_min = np.array([[self.params["min_v"]], [self.params["min_w"]]])
        self.du_max = np.array([[self.params["max_v_inc"]], [self.params["max_w_inc"]]])
    
    def __str__(self) -> str:
        return "Model Predicted Control (MPC)"

    def plan(self, path: list):
        """
        MPC motion plan function.

        Parameters:
            path (list): planner path from path planning module

        Returns:
            flag (bool): planning successful if true else failed
            real_path (list): real tracking path
            cost (float): real path cost
            pose_list (list): history poses of robot
        """
        u_p, lookahead_pts = DiffCmd(0, 0), []
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
                    u, u_p = self.mpcControl(robot_pose, lookahead_pt, u_r, u_p)

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

    def mpcControl(self, s: Point3d, s_d: Point3d, u_r: DiffCmd, u_p: DiffCmd) -> DiffCmd:
        """
        Execute MPC control process.

        Parameters:
            s (Point3d): current state
            s_d (Point3d): desired state
            u_r (DiffCmd): refered control
            u_p (DiffCmd): previous control error

        Returns:
            u (DiffCmd): control vector
        """
        dim_u, dim_x = 2, 3
        dt = self.params["time_step"]

        # state vector (5 x 1)
        x = np.array([
            [s.x() - s_d.x()],
            [s.y() - s_d.y()],
            [s.theta() - s_d.theta()],
            [u_p.v],
            [u_p.w],
        ])

        # original state matrix
        A = np.identity(3)
        A[0, 2] = -u_r.v * np.sin(s_d.theta()) * dt
        A[1, 2] = u_r.v * np.cos(s_d.theta()) * dt

        # original control matrix
        B = np.zeros((3, 2))
        B[0, 0] = np.cos(s_d.theta()) * dt
        B[1, 0] = np.sin(s_d.theta()) * dt
        B[2, 1] = dt

        # state matrix (5 x 5)
        A = np.concatenate((A, B), axis=1)
        temp = np.concatenate((np.zeros((dim_u, dim_x)), np.identity(dim_u)), axis=1)
        A = np.concatenate((A, temp), axis=0)

        # control matrix (5 x 2)
        B = np.concatenate((B, np.identity(dim_u)), axis=0)

        # output matrix (3 x 5)
        C = np.concatenate((np.identity(dim_x), np.zeros((dim_x, dim_u))), axis=1)

        # mpc state matrix (3p x 5)
        S_x = C @ A
        for i in range(1, self.p):
            S_x = np.concatenate((S_x, C @ np.linalg.matrix_power(A, i + 1)), axis=0)
        
        # mpc control matrix (3p x 2m)
        S_u_rows = []
        for i in range(self.p):
            S_u_row = C @ np.linalg.matrix_power(A, i) @ B
            for j in range(1, self.m):
                if j <= i:
                    S_u_row = np.concatenate((
                        S_u_row, C @ np.linalg.matrix_power(A, i - j) @ B
                    ), axis=1)
                else:
                    S_u_row = np.concatenate((S_u_row, np.zeros((dim_x, dim_u))), axis=1)
            S_u_rows.append(S_u_row)
        S_u = np.vstack(S_u_rows)

        # optimization
        Yr = np.zeros((3 * self.p, 1))              # (3p x 1)
        Q = np.kron(np.identity(self.p), self.Q)    # (3p x 3p)
        R = np.kron(np.identity(self.m), self.R)    # (2m x 2m)
        H = S_u.T @ Q @ S_u + R                     # (2m x 2m)
        g = S_u.T @ Q @ (S_x @ x - Yr)              # (2m x 1)

        # constriants
        I = np.eye(2 * self.m)
        A_I = np.kron(np.tril(np.ones((self.m, self.m))), np.diag([1, 1]))
        U_min = np.kron(np.ones((self.m, 1)), self.u_min)
        U_max = np.kron(np.ones((self.m, 1)), self.u_max)
        U_k_1 = np.kron(np.ones((self.m, 1)), np.array([[u_p.v], [u_p.w]]))

        # boundary
        dU_min = np.kron(np.ones((self.m, 1)), self.du_min)
        dU_max = np.kron(np.ones((self.m, 1)), self.du_max)

        # solve
        solver = osqp.OSQP()
        H = sparse.csc_matrix(H)
        A = sparse.csc_matrix(np.vstack([A_I, I]))
        l = np.vstack([U_min - U_k_1, dU_min])
        u = np.vstack([U_max - U_k_1, dU_max])
        solver.setup(H, g, A, l, u, verbose=False)
        res = solver.solve()
        dU_opt = res.x[:, None]
        
        # first element
        du = dU_opt[0:2]
        
        # real control
        u = du + np.array([[u_p.v], [u_p.w]]) + np.array([[u_r.v], [u_r.w]])

        return DiffCmd(
            self.linearRegularization(float(u[0])),
            self.angularRegularization(float(u[1]))
        ), DiffCmd(float(u[0]) - u_r.v, float(u[1]) - u_r.w)
