"""
@file: apf.py
@breif: Artificial Potential Field(APF) motion planning
@author: Winter
@update: 2023.10.24
"""
import math
import numpy as np
from scipy.spatial.distance import cdist

from .controller import Controller

from python_motion_planning.common.utils import LOG
from python_motion_planning.common.geometry import Point3d
from python_motion_planning.common.structure import DiffRobot, DiffCmd

class APF(Controller):
    """
    Class for Artificial Potential Field(APF) motion planning.

    Parameters:
        params (dict): parameters
    """
    def __init__(self, params: dict) -> None:
        super().__init__(params)
        # common
        self.start, self.goal, self.robot = None, None, None
        # APF parameters
        self.zeta = self.params["zeta"]
        self.eta = self.params["eta"]
        self.d_0 = self.params["d_0"]

    def __str__(self) -> str:
        return "Artificial Potential Field(APF)"
    
    def plan(self, path: list):
        """
        APF motion plan function.

        Parameters:
            path (list): planner path from path planning module

        Returns:
            flag (bool): planning successful if true else failed
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
            
            # compute the tatget pose and force at the current step
            lookahead_pt, _ = self.getLookaheadPoint(path)
            rep_force = self.getRepulsiveForce()
            attr_force = self.getAttractiveForce(robot_pose, lookahead_pt)
            net_force = self.zeta * attr_force + self.eta * rep_force
            
            # compute desired velocity
            v, theta = self.robot.v, self.robot.theta
            new_v = np.array([v * math.cos(theta), v * math.sin(theta)]) + net_force
            new_v /= np.linalg.norm(new_v)
            new_v *= self.params["max_v"]
            theta_d = math.atan2(new_v[1], new_v[0])

            # calculate velocity command
            e_theta = self.regularizeAngle(self.robot.theta - self.goal[2]) / 10
            if self.shouldRotateToGoal(robot_pose, self.goal):
                if not self.shouldRotateToPath(abs(e_theta)):
                    u = DiffCmd(0, 0)
                else:
                    u = DiffCmd(0, self.angularRegularization(e_theta / dt))
            else:
                e_theta = self.regularizeAngle(theta_d - self.robot.theta) / 10
                if self.shouldRotateToPath(abs(e_theta), np.pi / 4):
                    u = DiffCmd(0, self.angularRegularization(e_theta / dt))
                else:
                    v_d = np.linalg.norm(new_v)
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
    
    def getRepulsiveForce(self) -> np.ndarray:
        """
        Get the repulsive force of APF.

        Returns:
            rep_force (np.ndarray): repulsive force of APF
        """
        obstacles = np.array(list(self.obstacles))
        cur_pos = np.array([[self.robot.px, self.robot.py]])
        D = cdist(obstacles, cur_pos)
        rep_force = (1 / D - 1 / self.d_0) * (1 / D) ** 2 * (cur_pos - obstacles)
        valid_mask = np.argwhere((1 / D - 1 / self.d_0) > 0)[:, 0]
        rep_force = np.sum(rep_force[valid_mask, :], axis=0)

        if not np.all(rep_force == 0):
            rep_force = rep_force / np.linalg.norm(rep_force)
        
        return rep_force
    
    def getAttractiveForce(self, cur_pos: Point3d, tgt_pos: Point3d) -> np.ndarray:
        """
        Get the attractive force of APF.

        Parameters:
            cur_pos (Point3d): current position of robot
            tgt_pos (Point3d): target position of robot

        Returns
            attr_force (np.ndarray): attractive force
        """
        attr_force = np.array([tgt_pos.x(), tgt_pos.y()]) - np.array([cur_pos.x(), cur_pos.y()])
        if not np.all(attr_force == 0):
            attr_force = attr_force / np.linalg.norm(attr_force)
        
        return attr_force