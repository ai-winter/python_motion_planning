"""
@file: rpp.py
@breif: Regulated Pure Pursuit (RPP) motion planning
@author: Winter
@update: 2023.1.25
"""
import math
import numpy as np
from scipy.spatial.distance import cdist

from .controller import Controller

from python_motion_planning.common.utils import LOG
from python_motion_planning.common.geometry import Point3d
from python_motion_planning.common.structure import DiffRobot, DiffCmd

class RPP(Controller):
    """
    Class for RPP motion planning.

    Parameters:
        params (dict): parameters
    """
    def __init__(self, params: dict) -> None:
        super().__init__(params)
        # common
        self.start, self.goal, self.robot = None, None, None
        # RPP parameters
        self.regulated_radius_min = self.params["regulated_radius_min"]
        self.scaling_dist = self.params["scaling_dist"]
        self.scaling_gain = self.params["scaling_gain"]

    def __str__(self) -> str:
        return "Regulated Pure Pursuit (RPP)"

    def plan(self, path: list):
        """
        RPP motion plan function.

        Parameters:
            path (list): planner path from path planning module

        Returns:
            flag (bool): planning successful if true else failed
            pose_list (list): history poses of robot
            lookahead_pts (list): history lookahead points
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
            lookahead_pt, _ = self.getLookaheadPoint(path)

            # get the tracking curvature with goalahead point
            lookahead_k = 2 * math.sin(
                self.angle(robot_pose, lookahead_pt) - self.robot.theta
            ) / self.lookahead_dist

            # calculate velocity command
            e_theta = self.regularizeAngle(self.robot.theta - self.goal[2]) / 10
            if self.shouldRotateToGoal(robot_pose, self.goal):
                if not self.shouldRotateToPath(abs(e_theta)):
                    u = DiffCmd(0, 0)
                else:
                    u = DiffCmd(0, self.angularRegularization(e_theta / dt))
            else:
                e_theta = self.regularizeAngle(self.angle(robot_pose, lookahead_pt) - self.robot.theta) / 10
                if self.shouldRotateToPath(abs(e_theta), np.pi / 4):
                    u = DiffCmd(0, self.angularRegularization(e_theta / dt))
                else:
                    # apply constraints
                    curv_vel = self.applyCurvatureConstraint(self.params["max_v"], lookahead_k)
                    cost_vel = self.applyObstacleConstraint(self.params["max_v"])
                    v_d = min(curv_vel, cost_vel)
                    u = DiffCmd(self.linearRegularization(v_d), self.angularRegularization(v_d * lookahead_k))
            
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

    def applyCurvatureConstraint(self, raw_linear_vel: float, curvature: float) -> float:
        """
        Applying curvature constraints to regularize the speed of robot turning.

        Parameters:
            raw_linear_vel (float): the raw linear velocity of robot
            curvature (float): the tracking curvature

        Returns:
            reg_vel (float): the regulated velocity
        """
        radius = abs(1.0 / curvature)
        if radius < self.regulated_radius_min:
            return raw_linear_vel * (radius / self.regulated_radius_min)
        else:
            return raw_linear_vel
    
    def applyObstacleConstraint(self, raw_linear_vel: float) -> float:
        """
        Applying obstacle constraints to regularize the speed of robot approaching obstacles.

        Parameters:
            raw_linear_vel (float): the raw linear velocity of robot

        Returns:
            reg_vel (float): the regulated velocity
        """
        obstacles = np.array(list(self.obstacles))
        D = cdist(obstacles, np.array([[self.robot.px, self.robot.py]]))
        obs_dist = np.min(D)
        if obs_dist < self.scaling_dist:
            return raw_linear_vel * self.scaling_gain * obs_dist / self.scaling_dist
        else:
            return raw_linear_vel