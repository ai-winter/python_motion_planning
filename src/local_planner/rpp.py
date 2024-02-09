'''
@file: rpp.py
@breif: Regulated Pure Pursuit (RPP) motion planning
@author: Winter
@update: 2023.1.25
'''
import math
import os, sys
import numpy as np
from scipy.spatial.distance import cdist

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../")))

from .local_planner import LocalPlanner
from src.utils import Env


class RPP(LocalPlanner):
    '''
    Class for RPP motion planning.

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
    >>> from src.utils import Grid
    >>> from src.local_planner import RPP
    >>> start = (5, 5, 0)
    >>> goal = (45, 25, 0)
    >>> env = Grid(51, 31)
    >>> planner = RPP(start, goal, env)
    >>> planner.run()
    '''
    def __init__(self, start: tuple, goal: tuple, env: Env, heuristic_type: str = "euclidean") -> None:
        super().__init__(start, goal, env, heuristic_type)
        # RPP parameters
        self.regulated_radius_min = 0.9
        self.scaling_dist = 0.6
        self.scaling_gain = 1.0

        # global planner
        g_start = (start[0], start[1])
        g_goal  = (goal[0], goal[1])
        self.g_planner = {"planner_name": "a_star", "start": g_start, "goal": g_goal, "env": env}
        self.path = self.g_path[::-1]

    def __str__(self) -> str:
        return "Regulated Pure Pursuit (RPP)"

    def plan(self):
        '''
        RPP motion plan function.

        Return
        ----------
        flag: bool
            planning successful if true else failed
        pose_list: list
            history poses of robot
        lookahead_pts: list
            history lookahead points
        '''
        lookahead_pts = []
        dt = self.params["TIME_STEP"]
        for _ in range(self.params["MAX_ITERATION"]):
            # break until goal reached
            if self.shouldRotateToGoal(self.robot.position, self.goal):
                return True, self.robot.history_pose, lookahead_pts

            # get the particular point on the path at the lookahead distance
            lookahead_pt, _, _ = self.getLookaheadPoint()

            # get the tracking curvature with goalahead point
            lookahead_k = 2 * math.sin(
                self.angle(self.robot.position, lookahead_pt) - self.robot.theta
            ) / self.lookahead_dist

            # calculate velocity command
            e_theta = self.regularizeAngle(self.robot.theta - self.goal[2]) / 10
            if self.shouldRotateToGoal(self.robot.position, self.goal):
                if not self.shouldRotateToPath(abs(e_theta)):
                    u = np.array([[0], [0]])
                else:
                    u = np.array([[0], [self.angularRegularization(e_theta / dt)]])
            else:
                e_theta = self.regularizeAngle(
                    self.angle(self.robot.position, lookahead_pt) - self.robot.theta
                ) / 10
                if self.shouldRotateToPath(abs(e_theta), np.pi / 4):
                    u = np.array([[0], [self.angularRegularization(e_theta / dt)]])
                else:
                    # apply constraints
                    curv_vel = self.applyCurvatureConstraint(self.params["MAX_V"], lookahead_k)
                    cost_vel = self.applyObstacleConstraint(self.params["MAX_V"])
                    v_d = min(curv_vel, cost_vel)
                    u = np.array([[self.linearRegularization(v_d)], [self.angularRegularization(v_d * lookahead_k)]])
            
            # update lookahead points
            lookahead_pts.append(lookahead_pt)

            # feed into robotic kinematic
            self.robot.kinematic(u, dt)
        
        return False, None, None

    def run(self):
        '''
        Running both plannig and animation.
        '''
        _, history_pose, lookahead_pts = self.plan()
        if not history_pose:
            raise ValueError("Path not found and planning failed!")

        path = np.array(history_pose)[:, 0:2]
        cost = np.sum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1, keepdims=True)))
        self.plot.plotPath(self.path, path_color="r", path_style="--")
        self.plot.animation(path, str(self), cost, history_pose=history_pose, lookahead_pts=lookahead_pts)

    def applyCurvatureConstraint(self, raw_linear_vel: float, curvature: float) -> float:
        '''
        Applying curvature constraints to regularize the speed of robot turning.

        Parameters
        ----------
        raw_linear_vel: float
            the raw linear velocity of robot
        curvature: float
            the tracking curvature

        Return
        ----------
        reg_vel: float
            the regulated velocity
        '''
        radius = abs(1.0 / curvature)
        if radius < self.regulated_radius_min:
            return raw_linear_vel * (radius / self.regulated_radius_min)
        else:
            return raw_linear_vel
    
    def applyObstacleConstraint(self, raw_linear_vel: float) -> float:
        '''
        Applying obstacle constraints to regularize the speed of robot approaching obstacles.

        Parameters
        ----------
        raw_linear_vel: float
            the raw linear velocity of robot

        Return
        ----------
        reg_vel: float
            the regulated velocity
        '''
        obstacles = np.array(list(self.obstacles))
        D = cdist(obstacles, np.array([[self.robot.px, self.robot.py]]))
        obs_dist = np.min(D)
        if obs_dist < self.scaling_dist:
            return raw_linear_vel * self.scaling_gain * obs_dist / self.scaling_dist
        else:
            return raw_linear_vel