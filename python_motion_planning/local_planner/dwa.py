"""
@file: dwa.py
@breif: Dynamic Window Approach(DWA) motion planning
@author: Winter
@update: 2023.3.2
"""
import numpy as np
from itertools import product
from scipy.spatial.distance import cdist

from .local_planner import LocalPlanner
from python_motion_planning.utils import Env


class DWA(LocalPlanner):
    """
    Class for Dynamic Window Approach(DWA) motion planning.

    Parameters:
        start (tuple): start point coordinate
        goal (tuple): goal point coordinate
        env (Env): environment
        heuristic_type (str): heuristic function type

    Examples:
        >>> from python_motion_planning.utils import Grid
        >>> from python_motion_planning.local_planner import DWA
        >>> start = (5, 5, 0)
        >>> goal = (45, 25, 0)
        >>> env = Grid(51, 31)
        >>> planner = DWA(start, goal, env)
        >>> planner.run()

    References:
        [1] The Dynamic Window Approach to Collision Avoidance.
    """
    def __init__(self, start: tuple, goal: tuple, env: Env, heuristic_type: str = "euclidean", **params) -> None:
        super().__init__(start, goal, env, heuristic_type, **params)
        # kinematic parameters
        kinematic = {}
        kinematic["V_MAX"]         = 2.0                #  maximum velocity [m/s]
        kinematic["W_MAX"]         = 150.0 * np.pi / 180 #  maximum rotation speed[rad/s]
        kinematic["V_ACC"]         = 0.2                #  acceleration [m/s^2]
        kinematic["W_ACC"]         = 60.0 * np.pi / 180 #  angular acceleration [rad/s^2]
        kinematic["V_RESOLUTION"]  = 0.01               #  velocity resolution [m/s]
        kinematic["W_RESOLUTION"]  = 0.1 * np.pi / 180  #  rotation speed resolution [rad/s]]
        # robot
        self.robot.setParameters(**kinematic)
        
        # evalution parameters
        self.eval_param = {
            "heading": 0.045,
            "distance": 0.1,
            "velocity": 0.1,
            "predict_time": 3.0,
            "dt": 0.1,
            "R": 5.0
        }
        # threshold
        self.max_iter = 2000
        self.max_dist = 1.5

    def __str__(self) -> str:
        return "Dynamic Window Approach(DWA)"

    def plan(self):
        """
        Dynamic Window Approach(DWA) motion plan function.
        """
        history_traj = []
        for _ in range(self.max_iter):
            # dynamic configure
            vr = self.calDynamicWin()
            eval_win, traj_win = self.evaluation(vr)
        
            # failed
            if not len(eval_win):
                break
            
            # update
            max_index = np.argmax(eval_win[:, -1])
            u = np.expand_dims(eval_win[max_index, 0:-1], axis=1)

            self.robot.kinematic(u, self.eval_param["dt"])
            history_traj.append(traj_win[max_index])

            # goal found
            if self.dist((self.robot.px, self.robot.py), self.goal) < self.max_dist:
                return True, history_traj, self.robot.history_pose

        return False, None, None

    def calDynamicWin(self) -> list:
        """
        Calculate dynamic window.

        Returns:
            v_reference (list): reference velocity
        """
        # hard margin
        vs = (0, self.robot.V_MAX, -self.robot.W_MAX, self.robot.W_MAX)
        # predict margin
        vd = (
            self.robot.v - self.robot.V_ACC * self.eval_param["dt"], 
            self.robot.v + self.robot.V_ACC * self.eval_param["dt"], 
            self.robot.w - self.robot.W_ACC * self.eval_param["dt"],
            self.robot.w + self.robot.W_ACC * self.eval_param["dt"]
        )

        # dynamic window
        v_tmp = np.array([vs, vd])
        # reference velocity
        vr = [
            float(np.max(v_tmp[:, 0])), float(np.min(v_tmp[:, 1])), 
            float(np.max(v_tmp[:, 2])), float(np.min(v_tmp[:, 3]))
        ]
        return vr

    def evaluation(self, vr):
        """
        Extract the path based on the CLOSED set.

        Parameters:
            closed_set (list): CLOSED set

        Returns:
            cost (float): the cost of planning path
            path (list): the planning path
        """
        v_start, v_end, w_start, w_end = vr
        v = np.linspace(v_start, v_end, num=int((v_end - v_start) / self.robot.V_RESOLUTION)).tolist()
        w = np.linspace(w_start, w_end, num=int((w_end - w_start) / self.robot.W_RESOLUTION)).tolist()
        
        eval_win, traj_win = [], []
        for v_, w_ in product(v, w):
            # trajectory prediction, consistent of poses
            traj = self.generateTraj(v_, w_)
            end_state = traj[-1].squeeze().tolist()
            
            # heading evaluation
            theta = self.angle((end_state[0], end_state[1]), self.goal)
            heading = np.pi - abs(theta - end_state[2])

            # obstacle evaluation
            D = cdist(np.array(tuple(self.obstacles)), traj[:, 0:2])
            distance = np.min(D) if np.array(D < 1.2).any() else self.eval_param["R"]

            # velocity evaluation
            velocity = abs(v_)
            
            eval_win.append((v_, w_, heading, distance, velocity))
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
        factor = np.array([[1, 0,                          0],
                           [0, 1,                          0],
                           [0, 0, self.eval_param["heading"]],
                           [0, 0, self.eval_param["distance"]],
                           [0, 0, self.eval_param["velocity"]]])

        return eval_win @ factor, traj_win

    def generateTraj(self, v, w):
        """
        Generate predict trajectory.

        Returns:
            v_reference (list): reference velocity
        """
        u = np.array([[v], [w]])
        state = self.robot.state
        time_steps = int(self.eval_param["predict_time"] / self.eval_param["dt"])
        
        traj = []
        for _ in range(time_steps):
            state = self.robot.lookforward(state, u, self.eval_param["dt"])
            traj.append(state)
        
        return np.array(traj).squeeze()

    def run(self):
        """
        Running both plannig and animation.
        """
        _, history_traj, history_pose = self.plan()

        if not history_pose:
            raise ValueError("Path not found and planning failed!")

        path = np.array(history_pose)[:, 0:2]
        cost = np.sum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1, keepdims=True)))
        
        self.plot.animation(path, str(self), cost, history_pose=history_pose, predict_path=history_traj)



