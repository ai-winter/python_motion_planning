'''
@file: apf.py
@breif: Artificial Potential Field(APF) motion planning
@author: Winter
@update: 2023.10.24
'''
import math
import os, sys
import numpy as np
from scipy.spatial.distance import cdist

sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))

from .pid import PID
from utils import Env

class APF(PID):
    '''
    Class for Artificial Potential Field(APF) motion planning.

    Parameters
    ----------
    start: tuple
        start point coordinate
    goal: tuple
        goal point coordinate
    env: Env
        environment
    heuristic_type: str
        heuristic function type, default is euclidean

    Examples
    ----------
    >>> from utils import Grid
    >>> from local_planner import APF
    >>> start = (5, 5)
    >>> goal = (45, 25)
    >>> env = Grid(51, 31)
    >>> planner = APF(start, goal, env)
    >>> planner.run()
    '''
    def __init__(self, start: tuple, goal: tuple, env: Env, heuristic_type: str = "euclidean") -> None:
        super().__init__(start, goal, env, heuristic_type)
        # APF parameters
        self.zeta = 1.0
        self.eta = 1.5
        self.d_0 = 1.5
        self.max_v = 0.8
        self.obstacles = np.array(list(self.obstacles))

    def __str__(self) -> str:
        return "Artificial Potential Field(APF)"
    
    def plan(self):
        '''
        APF motion plan function.

        Return
        ----------
        flag: bool
            planning successful if true else failed
        pose_list: list
            history poses of robot
        '''
        plan_idx = 0
        for _ in range(self.max_iter):
            # break until goal reached
            if math.hypot(self.robot.px - self.goal[0], self.robot.py - self.goal[1]) < self.p_precision:
                return True, self.robot.history_pose
            
            #  compute the tatget pose and force at the current step
            rep_force = self.getRepulsiveForce()
            while plan_idx < len(self.path):
                tgt_pos = np.array(self.path[plan_idx])
                cur_pos = np.array([self.robot.px, self.robot.py])
                attr_force = self.getAttractiveForce(cur_pos, tgt_pos)
                net_force = self.zeta * attr_force + self.eta * rep_force

                # in body frame
                b_x_d = self.path[plan_idx][0] - self.robot.px
                b_y_d = self.path[plan_idx][1] - self.robot.py

                if math.hypot(b_x_d, b_y_d) > self.p_window:
                    break

                plan_idx += 1

            v, theta = self.robot.v, self.robot.theta
            new_v = np.array([v * math.cos(theta), v * math.sin(theta)]) + net_force
            new_v /= np.linalg.norm(new_v)
            new_v *= self.max_v
    
            theta_d = math.atan2(new_v[1], new_v[0])

            # calculate velocity command
            if math.hypot(self.robot.px - self.goal[0], self.robot.py - self.goal[1]) < self.p_precision:
                if abs(self.robot.theta - self.goal[2]) < self.o_precision:
                    u = np.array([[0], [0]])
                else:
                    u = np.array([[0], [self.angularController(self.goal[2])]])
            elif abs(theta_d - self.robot.theta) > np.pi / 2:
                u = np.array([[0], [self.angularController(theta_d)]])
            else:
                v_d = np.linalg.norm(new_v)
                u = np.array([[self.linearController(v_d)], [self.angularController(theta_d)]])
            
            # feed into robotic kinematic
            self.robot.kinematic(u, self.dt)
        
        return False, None
    
    def getRepulsiveForce(self) -> np.ndarray:
        '''
        Get the repulsive  force of APF.

        Return
        ----------
        rep_force: np.ndarray
            the repulsive force of APF
        '''
        cur_pos = np.array([[self.robot.px, self.robot.py]])
        D = cdist(self.obstacles, cur_pos)
        rep_force = (1 / D - 1 / self.d_0) * (1 / D) ** 2 * (cur_pos - self.obstacles)
        valid_mask = np.argwhere((1 / D - 1 / self.d_0) > 0)[:, 0]
        rep_force = np.sum(rep_force[valid_mask, :], axis=0)

        if not np.all(rep_force == 0):
            rep_force = rep_force / np.linalg.norm(rep_force)
        
        return rep_force
    
    def getAttractiveForce(self, cur_pos: np.ndarray, tgt_pos: np.ndarray) -> np.ndarray:
        '''
        Get the attractive force of APF.

        Parameters
        ----------
        cur_pos: np.ndarray
            current position of robot
        tgt_pos: np.ndarray
            target position of robot

        Return
        ----------
        attr_force: np.ndarray
            the attractive force
        '''
        attr_force = tgt_pos - cur_pos
        if not np.all(attr_force == 0):
            attr_force = attr_force / np.linalg.norm(attr_force)
        
        return attr_force