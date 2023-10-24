'''
@file: pid.py
@breif: PID motion planning
@author: Winter
@update: 2023.10.24
'''
import math
import os, sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))

from .local_planner import LocalPlanner
from utils import Env, Robot

class PID(LocalPlanner):
    '''
    Class for PID motion planning.

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
    '''
    def __init__(self, start: tuple, goal: tuple, env: Env, heuristic_type: str = "euclidean") -> None:
        super().__init__(start, goal, env, heuristic_type)
        # PID parameters
        self.dt = 0.1
        self.p_window = 1.0
        self.o_window = np.pi / 2
        self.p_precision = 0.5
        self.o_precision = np.pi / 4
        self.max_iter = 1000

        self.e_w, self.i_w = 0.0, 0.0
        self.e_v, self.i_v = 0.0, 0.0

        # robot
        self.robot = Robot(start[0], start[1], start[2], 0, 0)

        # global planner
        g_start = (start[0], start[1])
        g_goal  = (goal[0], goal[1])
        self.g_planner = {"planner_name": "a_star", "start": g_start, "goal": g_goal, "env": env}
        self.path = self.g_path[::-1]

    def __str__(self) -> str:
        return "PID Planner"

    def plan(self):
        '''
        PID motion plan function.

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
            
            # find next tracking point
            while plan_idx < len(self.path):
                theta_err = math.atan2(
                    self.path[plan_idx][1] - self.robot.py, 
                    self.path[plan_idx][0] - self.robot.px
                )

                if plan_idx < len(self.path) - 1:
                    theta_trj = math.atan2(
                        self.path[plan_idx + 1][1] - self.path[plan_idx][1],
                        self.path[plan_idx + 1][0] - self.path[plan_idx][0]
                    )

                if abs(theta_err - theta_trj) > np.pi:
                    if theta_err > theta_trj:
                        theta_trj += 2 * np.pi
                    else:
                        theta_err += 2 * np.pi

                k_theta = 0.5
                theta_d = k_theta * theta_err + (1 - k_theta) * theta_trj
                
                # in body frame
                b_x_d = self.path[plan_idx][0] - self.robot.px
                b_y_d = self.path[plan_idx][1] - self.robot.py
                b_theta_d = theta_d - self.robot.theta

                if math.hypot(b_x_d, b_y_d) > self.p_window or abs(b_theta_d) > self.o_window:
                    break

                plan_idx += 1
    
            # calculate velocity command
            if math.hypot(self.robot.px - self.goal[0], self.robot.py - self.goal[1]) < self.p_precision:
                if abs(self.robot.theta - self.goal[2]) < self.o_precision:
                    u = np.array([[0], [0]])
                else:
                    u = np.array([[0], [self.angularController(self.goal[2])]])
            elif abs(theta_d - self.robot.theta) > np.pi / 2:
                u = np.array([[0], [self.angularController(theta_d)]])
            else:
                v_d = math.hypot(b_x_d, b_y_d) / self.dt / 10
                u = np.array([[self.linearController(v_d)], [self.angularController(theta_d)]])

            # feed into robotic kinematic
            self.robot.kinematic(u, self.dt)
        
        return False, None

    def run(self):
        '''
        Running both plannig and animation.
        '''
        _, history_pose = self.plan()
        if not history_pose:
            raise ValueError("Path not found and planning failed!")

        path = np.array(history_pose)[:, 0:2]
        cost = np.sum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1, keepdims=True)))
        self.plot.plotPath(self.path, path_color="r", path_style="--")
        self.plot.animation(path, str(self), cost, history_pose=history_pose)

    def linearController(self, v_d: float) -> float:
        '''
        Linear velocity controller with pid.

        Parameters
        ----------
        v_d: float
            reference velocity input

        Return
        ----------
        v: float
            control velocity output
        '''
        e_v = v_d - self.robot.v
        self.i_v += e_v * self.dt
        d_v = (e_v - self.e_v) / self.dt
        self.e_v = e_v

        k_v_p = 1.00
        k_v_i = 0.00
        k_v_d = 0.00
        v_inc = k_v_p * e_v + k_v_i * self.i_v + k_v_d * d_v

        v = self.robot.v + v_inc
        
        return v

    def angularController(self, theta_d: float) -> float:
        '''
        Angular velocity controller with pid.

        Parameters
        ----------
        theta_d: float
            reference angular input

        Return
        ----------
        w: float
            control angular velocity output
        '''
        e_theta = self.regularizeAngle(theta_d - self.robot.theta)

        w_d = e_theta / self.dt / 10
        e_w = w_d - self.robot.w
        self.i_w += e_w * self.dt
        d_w = (e_w - self.e_w) / self.dt
        self.e_w = e_w

        k_w_p = 1.00
        k_w_i = 0.00
        k_w_d = 0.01
        w_inc = k_w_p * e_w + k_w_i * self.i_w + k_w_d * d_w

        w = self.robot.w + w_inc

        return w