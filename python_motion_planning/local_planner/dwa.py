"""
@file: dwa.py
@breif: Dynamic Window Approach(DWA) motion planning
@author: Yang Haodong, Wu Maojia
@update: 2024.6.25
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
        heading_weight (float): weight for heading cost
        obstacle_weight (float): weight for obstacle cost
        velocity_weight (float): weight for velocity cost
        predict_time (float): predict time for trajectory
        obstacle_inflation_radius (float): inflation radius for obstacles
        v_resolution (float): velocity resolution in evaulation
        w_resolution (float): angular velocity resolution in evaulation
        **params: other parameters can be found in the parent class LocalPlanner

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
    def __init__(self, start: tuple, goal: tuple, env: Env, heuristic_type: str = "euclidean",
                 heading_weight: float = 0.2, obstacle_weight: float = 0.1, velocity_weight: float = 0.05,
                 predict_time: float = 1.5, obstacle_inflation_radius: float = 1.0,
                 v_resolution: float = 0.05, w_resolution: float = 0.05, **params) -> None:
        super().__init__(start, goal, env, heuristic_type, **params)
        self.heading_weight = heading_weight
        self.obstacle_weight = obstacle_weight
        self.velocity_weight = velocity_weight
        self.predict_time = predict_time
        self.obstacle_inflation_radius = obstacle_inflation_radius
        self.v_resolution = v_resolution
        self.w_resolution = w_resolution

        # global planner
        g_start = (start[0], start[1])
        g_goal  = (goal[0], goal[1])
        self.g_planner = {"planner_name": "a_star", "start": g_start, "goal": g_goal, "env": env}
        self.path = self.g_path[::-1]

    def __str__(self) -> str:
        return "Dynamic Window Approach(DWA)"

    def plan(self) -> tuple:
        """
        Dynamic Window Approach(DWA) motion plan function.
        """
        history_traj = []
        for _ in range(self.params["MAX_ITERATION"]):
            # break until goal reached
            if self.reachGoal(tuple(self.robot.state.squeeze(axis=1)[0:3]), self.goal):
                return True, history_traj, self.robot.history_pose

            # get the particular point on the path at the lookahead distance to track
            lookahead_pt, theta_trj, kappa = self.getLookaheadPoint()

            # dynamic configure
            vr = self.calDynamicWin()
            eval_win, traj_win = self.evaluation(vr, lookahead_pt)

            # failed
            if not len(eval_win):
                break

            # update
            max_index = np.argmax(eval_win[:, -1])
            u = np.expand_dims(eval_win[max_index, 0:-1], axis=1)

            self.robot.kinematic(u, self.params["TIME_STEP"])
            history_traj.append(traj_win[max_index])

        return False, None, None

    def run(self) -> None:
        """
        Running both plannig and animation.
        """
        _, history_traj, history_pose = self.plan()

        if not history_pose:
            raise ValueError("Path not found and planning failed!")

        path = np.array(history_pose)[:, 0:2]
        cost = np.sum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1, keepdims=True)))
        self.plot.plotPath(self.path, path_color="r", path_style="--")
        self.plot.animation(path, str(self), cost, history_pose=history_pose, predict_path=history_traj)

    def calDynamicWin(self) -> list:
        """
        Calculate dynamic window.

        Returns:
            v_reference (list): reference velocity
        """
        # hard margin
        vs = (self.params["MIN_V"], self.params["MAX_V"], self.params["MIN_W"], self.params["MAX_W"])
        # predict margin
        vd = (
            self.robot.v + self.params["MIN_V_INC"] * self.params["TIME_STEP"],
            self.robot.v + self.params["MAX_V_INC"] * self.params["TIME_STEP"],
            self.robot.w +self.params["MIN_W_INC"] * self.params["TIME_STEP"],
            self.robot.w + self.params["MAX_W_INC"] * self.params["TIME_STEP"]
        )

        # dynamic window
        v_tmp = np.array([vs, vd])
        # reference velocity
        vr = [
            float(np.max(v_tmp[:, 0])), float(np.min(v_tmp[:, 1])), 
            float(np.max(v_tmp[:, 2])), float(np.min(v_tmp[:, 3]))
        ]
        return vr

    def evaluation(self, vr: list, goal: tuple) -> tuple:
        """
        Extract the path based on the CLOSED set.

        Parameters:
            closed_set (list): CLOSED set
            goal (tuple): goal point coordinate

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
            theta = self.angle((end_state[0], end_state[1]), goal)
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
        u = np.array([[v], [w]])
        state = self.robot.state
        time_steps = int(self.predict_time / self.params["TIME_STEP"])
        
        traj = []
        for _ in range(time_steps):
            state = self.robot.lookforward(state, u, self.params["TIME_STEP"])
            traj.append(state)
        
        return np.array(traj).squeeze()
