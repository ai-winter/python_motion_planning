"""
@file: pid.py
@breif: PID motion planning
@author: Winter
@update: 2023.10.24
"""
import numpy as np

from .local_planner import LocalPlanner
from python_motion_planning.utils import Env


class PID(LocalPlanner):
    """
    Class for PID motion planning.

    Parameters:
        start (tuple): start point coordinate
        goal (tuple): goal point coordinate
        env (Env): environment
        heuristic_type (str): heuristic function type

    Examples:
        >>> from python_motion_planning.utils import Grid
        >>> from python_motion_planning.local_planner import PID
        >>> start = (5, 5, 0)
        >>> goal = (45, 25, 0)
        >>> env = Grid(51, 31)
        >>> planner = PID(start, goal, env)
        >>> planner.run()
    """
    def __init__(self, start: tuple, goal: tuple, env: Env, heuristic_type: str = "euclidean") -> None:
        super().__init__(start, goal, env, heuristic_type)
        # PID parameters
        self.e_w, self.i_w = 0.0, 0.0
        self.e_v, self.i_v = 0.0, 0.0

        # global planner
        g_start = (start[0], start[1])
        g_goal  = (goal[0], goal[1])
        self.g_planner = {"planner_name": "a_star", "start": g_start, "goal": g_goal, "env": env}
        self.path = self.g_path[::-1]

    def __str__(self) -> str:
        return "PID Planner"

    def plan(self):
        """
        PID motion plan function.

        Returns:
            flag (bool): planning successful if true else failed
            pose_list (list): history poses of robot
        """
        dt = self.params["TIME_STEP"]
        for _ in range(self.params["MAX_ITERATION"]):
            # break until goal reached
            if self.shouldRotateToGoal(self.robot.position, self.goal):
                return True, self.robot.history_pose
            
            # find next tracking point
            lookahead_pt, theta_trj, _ = self.getLookaheadPoint()

            # desired angle
            k_theta = 0.5
            theta_err = self.angle(self.robot.position, lookahead_pt)
            if abs(theta_err - theta_trj) > np.pi:
                if theta_err > theta_trj:
                    theta_trj += 2 * np.pi
                else:
                    theta_err += 2 * np.pi
            theta_d = k_theta * theta_err + (1 - k_theta) * theta_trj
    
            # calculate velocity command
            e_theta = self.regularizeAngle(self.robot.theta - self.goal[2]) / 10
            if self.shouldRotateToGoal(self.robot.position, self.goal):
                if not self.shouldRotateToPath(abs(e_theta)):
                    u = np.array([[0], [0]])
                else:
                    u = np.array([[0], [self.angularRegularization(e_theta / dt)]])
            else:
                e_theta = self.regularizeAngle(theta_d - self.robot.theta) / 10
                if self.shouldRotateToPath(abs(e_theta), np.pi / 4):
                    u = np.array([[0], [self.angularRegularization(e_theta / dt)]])
                else:
                    v_d = self.dist(lookahead_pt, self.robot.position) / dt / 10
                    u = np.array([[self.linearRegularization(v_d)], [self.angularRegularization(e_theta / dt)]])

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

    def linearRegularization(self, v_d: float) -> float:
        """
        Linear velocity controller with pid.

        Parameters:
            v_d (float): reference velocity input

        Returns:
            v (float): control velocity output
        """
        e_v = v_d - self.robot.v
        self.i_v += e_v * self.params["TIME_STEP"]
        d_v = (e_v - self.e_v) / self.params["TIME_STEP"]
        self.e_v = e_v

        k_v_p = 1.00
        k_v_i = 0.00
        k_v_d = 0.00
        v_inc = k_v_p * e_v + k_v_i * self.i_v + k_v_d * d_v

        v = self.robot.v + v_inc
        
        return v

    def angularRegularization(self, w_d: float) -> float:
        """
        Angular velocity controller with pid.

        Parameters:
            w_d (float): reference angular input

        Returns:
            w (float): control angular velocity output
        """
        e_w = w_d - self.robot.w
        self.i_w += e_w * self.params["TIME_STEP"]
        d_w = (e_w - self.e_w) / self.params["TIME_STEP"]
        self.e_w = e_w

        k_w_p = 1.00
        k_w_i = 0.00
        k_w_d = 0.01
        w_inc = k_w_p * e_w + k_w_i * self.i_w + k_w_d * d_w

        w = self.robot.w + w_inc

        return w