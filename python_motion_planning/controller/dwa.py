"""
@file: dwa.py
@breif: Dynamic Window Approach(DWA) motion planning
@author: Yang Haodong
@update: 2024.6.25
"""
import numpy as np
from itertools import product
from scipy.spatial.distance import cdist

from .controller import Controller

from python_motion_planning.common.utils import LOG
from python_motion_planning.common.geometry import Point3d
from python_motion_planning.common.structure import DiffRobot, DiffCmd

class DWA(Controller):
    """
    Class for Dynamic Window Approach(DWA) motion planning.

    Parameters:
        params (dict): parameters

    References:
        [1] The Dynamic Window Approach to Collision Avoidance.
    """
    def __init__(self, params: dict) -> None:
        super().__init__(params)
        # common
        self.start, self.goal, self.robot = None, None, None
        # DWA parameters
        self.heading_weight = self.params["heading_weight"]
        self.obstacle_weight = self.params["obstacle_weight"]
        self.velocity_weight = self.params["velocity_weight"]
        self.predict_time = self.params["predict_time"]
        self.obstacle_inflation_radius = self.params["obstacle_inflation_radius"]
        self.v_resolution = self.params["v_resolution"]
        self.w_resolution = self.params["w_resolution"]

    def __str__(self) -> str:
        return "Dynamic Window Approach(DWA)"

    def plan(self, path: list) -> tuple:
        """
        DWA motion plan function.

        Parameters:
            path (list): planner path from path planning module

        Returns:
            flag (bool): planning successful if true else failed
            real_path (list): real tracking path
            cost (float): real path cost
            pose_list (list): history poses of robot
        """
        history_traj, lookahead_pts = [], []
        self.start, self.goal = path[0], path[-1]
        self.robot = DiffRobot(self.start.x(), self.start.y(), self.start.theta(), 0, 0)
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
                    {"type": "frame", "name": "line", "props": {"color": "#ddd"},
                     "data": [[[traj[:, 0], traj[:, 1]] for traj in trajs] for trajs in history_traj]},
                ]

            # get the particular point on the path at the lookahead distance to track
            lookahead_pt, _ = self.getLookaheadPoint(path)

            # dynamic configure
            vr = self.calDynamicWin()
            eval_win, traj_win = self.evaluation(vr, lookahead_pt)

            # failed
            if not len(eval_win):
                break

            # update
            max_index = np.argmax(eval_win[:, -1])
            u = eval_win[max_index, 0:-1]
            u = DiffCmd(u[0], u[1])
            lookahead_pts.append(lookahead_pt)
            self.robot.kinematic(u, self.params["time_step"])
            history_traj.append(traj_win)

        LOG.WARN(f"{str(self)} Controller Controlling Failed!")
        return [
            {"type": "value", "data": False, "name": "success"},
            {"type": "value", "data": 0, "name": "cost"},
            {"type": "path", "data": [], "name": "normal"},
            {"type": "frame", "data": [], "name": "agent"},
            {"type": "frame", "data": [], "name": "marker"},
            {"type": "frame", "data": [], "name": "line"},
            {"type": "frame", "data": [], "name": "line"},
        ]

    def calDynamicWin(self) -> list:
        """
        Calculate dynamic window.

        Returns:
            v_reference (list): reference velocity
        """
        # hard margin
        vs = (self.params["min_v"], self.params["max_v"], self.params["min_w"], self.params["max_w"])
        # predict margin
        vd = (
            self.robot.v + self.params["min_v_inc"] * self.params["time_step"],
            self.robot.v + self.params["max_v_inc"] * self.params["time_step"],
            self.robot.w +self.params["min_w_inc"] * self.params["time_step"],
            self.robot.w + self.params["max_w_inc"] * self.params["time_step"]
        )

        # dynamic window
        v_tmp = np.array([vs, vd])
        # reference velocity
        vr = [
            float(np.max(v_tmp[:, 0])), float(np.min(v_tmp[:, 1])), 
            float(np.max(v_tmp[:, 2])), float(np.min(v_tmp[:, 3]))
        ]
        return vr

    def evaluation(self, vr: list, goal: Point3d) -> tuple:
        """
        Extract the path based on the CLOSED set.

        Parameters:
            closed_set (list): CLOSED set
            goal (Point3d): goal point coordinate

        Returns:
            cost (float): the cost of planning path
            path (list): the planning path
        """
        v_start, v_end, w_start, w_end = vr
        v = np.linspace(v_start, v_end, num=int((v_end - v_start) / self.v_resolution)).tolist()
        w = np.linspace(w_start, w_end, num=int((w_end - w_start) / self.w_resolution)).tolist()
        
        eval_win, traj_win = [], []
        for v_, w_ in product(v, w):
            # trajectory prediction, consistent of poses
            traj = self.generateTraj(v_, w_)
            end_state = traj[-1].squeeze().tolist()
            
            # heading evaluation
            theta = self.angle(Point3d(end_state[0], end_state[1]), goal)
            heading = np.pi - abs(theta - end_state[2])

            # obstacle evaluation
            D = cdist(np.array(tuple(self.obstacles)), traj[:, 0:2])
            min_D = np.min(D)
            obstacle = min(min_D, self.obstacle_inflation_radius)

            # velocity evaluation
            velocity = abs(v_)
            
            eval_win.append((v_, w_, heading, obstacle, velocity))
            traj_win.append(traj)
    
        # normalization
        eval_win = np.array(eval_win)
        if np.sum(eval_win[:, 2]) != 0:
            eval_win[:, 2] = eval_win[:, 2] / np.sum(eval_win[:, 2])
        if np.sum(eval_win[:, 3]) != 0:
            eval_win[:, 3] = eval_win[:, 3] / np.sum(eval_win[:, 3])
        if np.sum(eval_win[:, 4]) != 0:
            eval_win[:, 4] = eval_win[:, 4] / np.sum(eval_win[:, 4])
        
        # evaluation
        factor = np.array([[1, 0,                    0],
                           [0, 1,                    0],
                           [0, 0, self.heading_weight ],
                           [0, 0, self.obstacle_weight],
                           [0, 0, self.velocity_weight]])

        return eval_win @ factor, traj_win

    def generateTraj(self, v: float, w: float) -> np.ndarray:
        """
        Generate predict trajectory.

        Parameters:
            v (float): velocity
            w (float): angular velocity

        Returns:
            v_reference (np.ndarray): reference velocity
        """
        u = DiffCmd(v, w)
        state = self.robot.state
        time_steps = int(self.predict_time / self.params["time_step"])
        
        traj = []
        for _ in range(time_steps):
            state = self.robot.lookforward(state, u, self.params["time_step"])
            traj.append(state)
        
        return np.array(traj).squeeze()
