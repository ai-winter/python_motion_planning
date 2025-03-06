"""
@file: apf.py
@breif: Artificial Potential Field(APF) motion planning
@author: Yang Haodong, Wu Maojia
@update: 2024.5.21
"""
import math
import numpy as np
from scipy.spatial.distance import cdist

from .local_planner import LocalPlanner
from python_motion_planning.utils import Env

class APF(LocalPlanner):
    """
    Class for Artificial Potential Field(APF) motion planning.

    Parameters:
        start (tuple): start point coordinate
        goal (tuple): goal point coordinate
        env (Env): environment
        heuristic_type (str): heuristic function type
        **params: other parameters can be found in the parent class LocalPlanner

    Examples:
        >>> from python_motion_planning.utils import Grid
        >>> from python_motion_planning.local_planner import APF
        >>> start = (5, 5, 0)
        >>> goal = (45, 25, 0)
        >>> env = Grid(51, 31)
        >>> planner = APF(start, goal, env)
        >>> planner.run()
    """
    def __init__(self, start: tuple, goal: tuple, env: Env, heuristic_type: str = "euclidean", **params) -> None:
        super().__init__(start, goal, env, heuristic_type, **params)
        # APF parameters
        self.zeta = 1.0
        self.eta = 1.0
        self.d_0 = 1.0

        # global planner
        g_start = (start[0], start[1])
        g_goal  = (goal[0], goal[1])
        self.g_planner = {"planner_name": "a_star", "start": g_start, "goal": g_goal, "env": env}
        self.path = self.g_path[::-1]

    def __str__(self) -> str:
        return "Artificial Potential Field(APF)"
    
    def plan(self):
        """
        APF motion plan function.

        Returns:
            flag (bool): planning successful if true else failed
            pose_list (list): history poses of robot
        """
        dt = self.params["TIME_STEP"]
        for _ in range(self.params["MAX_ITERATION"]):
            # break until goal reached
            if self.reachGoal(tuple(self.robot.state.squeeze(axis=1)[0:3]), self.goal):
                return True, self.robot.history_pose
            
            # compute the tatget pose and force at the current step
            lookahead_pt, _, _ = self.getLookaheadPoint()
            rep_force = self.getRepulsiveForce()
            attr_force = self.getAttractiveForce(np.array(self.robot.position), np.array(lookahead_pt))
            net_force = self.zeta * attr_force + self.eta * rep_force
            
            # compute desired velocity
            v, theta = self.robot.v, self.robot.theta
            new_v = np.array([v * math.cos(theta), v * math.sin(theta)]) + net_force
            new_v /= np.linalg.norm(new_v)
            new_v *= self.params["MAX_V"]
            theta_d = math.atan2(new_v[1], new_v[0])

            # calculate velocity command
            e_theta = self.regularizeAngle(self.robot.theta - self.goal[2])
            if not self.shouldMoveToGoal(self.robot.position, self.goal):
                if not self.shouldRotateToPath(abs(e_theta)):
                    u = np.array([[0], [0]])
                else:
                    u = np.array([[0], [self.angularRegularization(e_theta / dt)]])
            else:
                e_theta = self.regularizeAngle(theta_d - self.robot.theta)
                if self.shouldRotateToPath(abs(e_theta)):
                    u = np.array([[0], [self.angularRegularization(e_theta / dt)]])
                else:
                    v_d = np.linalg.norm(new_v)
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
    
    def getAttractiveForce(self, cur_pos: np.ndarray, tgt_pos: np.ndarray) -> np.ndarray:
        """
        Get the attractive force of APF.

        Parameters:
            cur_pos (np.ndarray): current position of robot
            tgt_pos (np.ndarray): target position of robot

        Returns
            attr_force (np.ndarray): attractive force
        """
        attr_force = tgt_pos - cur_pos
        if not np.all(attr_force == 0):
            attr_force = attr_force / np.linalg.norm(attr_force)
        
        return attr_force