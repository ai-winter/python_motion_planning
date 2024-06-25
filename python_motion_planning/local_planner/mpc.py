"""
@file: mpc.py
@breif: Model Predicted Control (MPC) motion planning
@author: Yang Haodong, Wu Maojia
@update: 2024.6.25
"""
import osqp
import numpy as np
from scipy import sparse

from .local_planner import LocalPlanner
from python_motion_planning.utils import Env

class MPC(LocalPlanner):
    """
    Class for Model Predicted Control (MPC) motion planning.

    Parameters:
        start (tuple): start point coordinate
        goal (tuple): goal point coordinate
        env (Env): environment
        heuristic_type (str): heuristic function type
        **params: other parameters can be found in the parent class LocalPlanner

    Examples:
        >>> from python_motion_planning.utils import Grid
        >>> from python_motion_planning.local_planner import MPC
        >>> start = (5, 5, 0)
        >>> goal = (45, 25, 0)
        >>> env = Grid(51, 31)
        >>> planner = MPC(start, goal, env)
        >>> planner.run()
    """
    def __init__(self, start: tuple, goal: tuple, env: Env, heuristic_type: str = "euclidean", **params) -> None:
        super().__init__(start, goal, env, heuristic_type, **params)
        # MPC parameters
        self.p = 12
        self.m = 8
        self.Q = np.diag([0.8, 0.8, 0.5])
        self.R = np.diag([2, 2])
        self.u_min = np.array([[self.params["MIN_V"]], [self.params["MIN_W"]]])
        self.u_max = np.array([[self.params["MAX_V"]], [self.params["MAX_W"]]])
        # self.du_min = np.array([[self.params["MIN_V"]], [self.params["MIN_W"]]])
        self.du_min = np.array([[self.params["MIN_V_INC"]], [self.params["MIN_W_INC"]]])
        self.du_max = np.array([[self.params["MAX_V_INC"]], [self.params["MAX_W_INC"]]])

        # global planner
        g_start = (start[0], start[1])
        g_goal  = (goal[0], goal[1])
        self.g_planner = {"planner_name": "a_star", "start": g_start, "goal": g_goal, "env": env}
        self.path = self.g_path[::-1]
    
    def __str__(self) -> str:
        return "Model Predicted Control (MPC)"

    def plan(self):
        """
        MPC motion plan function.

        Returns:
            flag (bool): planning successful if true else failed
            pose_list (list): history poses of robot
        """
        dt = self.params["TIME_STEP"]
        u_p = (0, 0)
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
                    u, u_p = self.mpcControl(s, s_d, u_r, u_p)

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

    def mpcControl(self, s: tuple, s_d: tuple, u_r: tuple, u_p: tuple) -> np.ndarray:
        """
        Execute MPC control process.

        Parameters:
            s (tuple): current state
            s_d (tuple): desired state
            u_r (tuple): refered control
            u_p (tuple): previous control error

        Returns:
            u (np.ndarray): control vector
        """
        dim_u, dim_x = 2, 3
        dt = self.params["TIME_STEP"]

        # state vector (5 x 1)
        x = np.array([
            [s[0] - s_d[0]],
            [s[1] - s_d[1]],
            [s[2] - s_d[2]],
            [u_p[0]],
            [u_p[1]],
        ])

        # original state matrix
        A = np.identity(3)
        A[0, 2] = -u_r[0] * np.sin(s_d[2]) * dt
        A[1, 2] = u_r[0] * np.cos(s_d[2]) * dt

        # original control matrix
        B = np.zeros((3, 2))
        B[0, 0] = np.cos(s_d[2]) * dt
        B[1, 0] = np.sin(s_d[2]) * dt
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
        U_k_1 = np.kron(np.ones((self.m, 1)), np.array([[u_p[0]], [u_p[1]]]))

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
        u = du + np.array([[u_p[0]], [u_p[1]]]) + np.array([[u_r[0]], [u_r[1]]])

        return np.array([
            [self.linearRegularization(float(u[0]))], 
            [self.angularRegularization(float(u[1]))]
        ]), (float(u[0]) - u_r[0], float(u[1]) - u_r[1])
