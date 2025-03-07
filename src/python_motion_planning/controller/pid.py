"""
@file: pid.py
@breif: PID motion planning
@author: Winter
@update: 2023.10.24
"""
import numpy as np

from .controller import Controller

from python_motion_planning.common.utils import LOG
from python_motion_planning.common.math import MathHelper
from python_motion_planning.common.geometry import Point3d
from python_motion_planning.common.structure import DiffRobot, DiffCmd

class PID(Controller):
    """
    Class for PID motion planning.

    Parameters:
        params (dict): parameters
    """
    def __init__(self, params: dict) -> None:
        super().__init__(params)
        # common
        self.start, self.goal, self.robot = None, None, None
        # PID parameters
        self.e_w, self.i_w = 0.0, 0.0
        self.e_v, self.i_v = 0.0, 0.0

    def __str__(self) -> str:
        return "PID Planner"

    def plan(self, path: list):
        """
        PID motion plan function.

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
            
            # find next tracking point
            lookahead_pt, _ = self.getLookaheadPoint(path)

            # desired angle
            k_theta, theta_trj = 0.5, lookahead_pt.theta()
            theta_err = self.angle(robot_pose, lookahead_pt)
            if abs(theta_err - theta_trj) > np.pi:
                if theta_err > theta_trj:
                    theta_trj += 2 * np.pi
                else:
                    theta_err += 2 * np.pi
            theta_d = k_theta * theta_err + (1 - k_theta) * theta_trj
    
            # calculate velocity command
            e_theta = self.regularizeAngle(self.robot.theta - self.goal[2])
            if self.shouldRotateToGoal(robot_pose, self.goal):
                if not self.shouldRotateToPath(abs(e_theta)):
                    u = DiffCmd(0, 0)
                else:
                    u = DiffCmd(0, self.angularRegularization(e_theta / dt))
            else:
                e_theta = self.regularizeAngle(theta_d - self.robot.theta)
                if self.shouldRotateToPath(abs(e_theta)):
                    u = DiffCmd(0, self.angularRegularization(e_theta / dt))
                else:
                    v_d = self.dist(lookahead_pt, robot_pose) / dt
                    u = DiffCmd(self.linearRegularization(v_d), self.angularRegularization(e_theta / dt))

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

    def linearRegularization(self, v_d: float) -> float:
        """
        Linear velocity controller with pid.

        Parameters:
            v_d (float): reference velocity input

        Returns:
            v (float): control velocity output
        """
        e_v = v_d - self.robot.v
        self.i_v += e_v * self.params["time_step"]
        d_v = (e_v - self.e_v) / self.params["time_step"]
        self.e_v = e_v

        k_v_p = 1.00
        k_v_i = 0.00
        k_v_d = 0.00
        v_inc = k_v_p * e_v + k_v_i * self.i_v + k_v_d * d_v
        v_inc = MathHelper.clamp(v_inc, self.params["min_v_inc"], self.params["max_v_inc"])

        v = self.robot.v + v_inc
        v = MathHelper.clamp(v, self.params["min_v"], self.params["max_v"])
        
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
        self.i_w += e_w * self.params["time_step"]
        d_w = (e_w - self.e_w) / self.params["time_step"]
        self.e_w = e_w

        k_w_p = 1.00
        k_w_i = 0.00
        k_w_d = 0.01
        w_inc = k_w_p * e_w + k_w_i * self.i_w + k_w_d * d_w
        w_inc = MathHelper.clamp(w_inc, self.params["min_w_inc"], self.params["max_w_inc"])

        w = self.robot.w + w_inc
        w = MathHelper.clamp(w, self.params["min_w"], self.params["max_w"])

        return w