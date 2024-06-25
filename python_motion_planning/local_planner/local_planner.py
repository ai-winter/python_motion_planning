"""
@file: local_planner.py
@breif: Base class for local planner.
@author: Yang Haodong, Wu Maojia
@update: 2024.5.20
"""
import math

from python_motion_planning.utils import Env, Planner, SearchFactory, Plot, Robot, MathHelper


class LocalPlanner(Planner):
    """
    Base class for local planner.

    Parameters:
        start (tuple): start point coordinate
        goal (tuple): goal point coordinate
        env (Env): environment
        heuristic_type (str): heuristic function type
        **params: other parameters
    """
    def __init__(self, start: tuple, goal: tuple, env: Env, heuristic_type: str="euclidean", **params) -> None:
        # start and goal pose
        assert len(start) == 3 and len(goal) == 3, \
            "Start and goal parameters must be (x, y, theta)"
        self.start, self.goal = start, goal
        # heuristic type
        self.heuristic_type = heuristic_type
        # environment
        self.env = env
        # obstacles
        self.obstacles = self.env.obstacles
        # graph handler
        self.plot = Plot(start, goal, env)
        # robot
        self.robot = Robot(start[0], start[1], start[2], 0, 0)

        # parameters and default value
        self.params = {}
        self.params["TIME_STEP"] = params["TIME_STEP"] if "TIME_STEP" in params.keys() else 0.1
        self.params["MAX_ITERATION"] = params["MAX_ITERATION"] if "MAX_ITERATION" in params.keys() else 1500
        self.params["LOOKAHEAD_TIME"] = params["LOOKAHEAD_TIME"] if "LOOKAHEAD_TIME" in params.keys() else 1.0
        self.params["MAX_LOOKAHEAD_DIST"] = params["MAX_LOOKAHEAD_DIST"] if "MAX_LOOKAHEAD_DIST" in params.keys() else 2.5
        self.params["MIN_LOOKAHEAD_DIST"] = params["MIN_LOOKAHEAD_DIST"] if "MIN_LOOKAHEAD_DIST" in params.keys() else 1.0
        self.params["MAX_V_INC"] = params["MAX_V_INC"] if "MAX_V_INC" in params.keys() else 1.0
        self.params["MIN_V_INC"] = params["MIN_V_INC"] if "MIN_V_INC" in params.keys() else -1.0
        self.params["MAX_V"] = params["MAX_V"] if "MAX_V" in params.keys() else 0.5
        self.params["MIN_V"] = params["MIN_V"] if "MIN_V" in params.keys() else 0.0
        self.params["MAX_W_INC"] = params["MAX_W_INC"] if "MAX_W_INC" in params.keys() else math.pi
        self.params["MIN_W_INC"] = params["MIN_W_INC"] if "MIN_W_INC" in params.keys() else -math.pi
        self.params["MAX_W"] = params["MAX_W"] if "MAX_W" in params.keys() else math.pi / 2
        self.params["MIN_W"] = params["MIN_W"] if "MIN_W" in params.keys() else -math.pi / 2
        self.params["GOAL_DIST_TOL"] = params["GOAL_DIST_TOL"] if "GOAL_DIST_TOL" in params.keys() else 0.5
        self.params["ROTATE_TOL"] = params["ROTATE_TOL"] if "ROTATE_TOL" in params.keys() else 0.5

        # global planner
        self.g_planner_ = None
        # global path
        self.path = None
        # search factory
        self.search_factory_ = SearchFactory()
    
    @property
    def g_planner(self):
        return str(self.g_planner_)
    
    @g_planner.setter
    def g_planner(self, config):
        if "planner_name" in config:
            self.g_planner_ = self.search_factory_(**config)
        else:
            raise RuntimeError("Please set planner name!")
    
    @property
    def g_path(self):
        """
        [property]Global path.
        """
        if self.g_planner_ is None:
            raise AttributeError("Global path searcher is None, please set it first!")
        
        cost, path, _ = self.g_planner_.plan()
        return path

    @property
    def lookahead_dist(self):
        return MathHelper.clamp(
            abs(self.robot.v) * self.params["LOOKAHEAD_TIME"],
            self.params["MIN_LOOKAHEAD_DIST"],
            self.params["MAX_LOOKAHEAD_DIST"]
        )

    def dist(self, start: tuple, end: tuple) -> float:
        return math.hypot(end[0] - start[0], end[1] - start[1])
    
    def angle(self, start: tuple, end: tuple) -> float:
        return math.atan2(end[1] - start[1], end[0] - start[0])

    def regularizeAngle(self, angle: float):
        return angle - 2.0 * math.pi * math.floor((angle + math.pi) / (2.0 * math.pi))

    def getLookaheadPoint(self):
        """
        Find the point on the path that is exactly the lookahead distance away from the robot

        Returns:
            lookahead_pt (tuple): lookahead point
            theta (float): the angle on trajectory
            kappa (float): the curvature on trajectory
        """
        if self.path is None:
            assert RuntimeError("Please plan the path using g_planner!")

        # Find the first pose which is at a distance greater than the lookahead distance
        dist_to_robot = [self.dist(p, (self.robot.px, self.robot.py)) for p in self.path]
        idx_closest = dist_to_robot.index(min(dist_to_robot))
        idx_goal = len(self.path) - 1
        idx_prev = idx_goal - 1
        for i in range(idx_closest, len(self.path)):
            if self.dist(self.path[i], (self.robot.px, self.robot.py)) >= self.lookahead_dist:
                idx_goal = i
                break

        pt_x, pt_y = None, None
        if idx_goal == len(self.path) - 1:
            # If the no pose is not far enough, take the last pose
            pt_x = self.path[idx_goal][0]
            pt_y = self.path[idx_goal][1]
        else:
            if idx_goal == 0:
                idx_goal = idx_goal + 1
            #  find the point on the line segment between the two poses
            #  that is exactly the lookahead distance away from the robot pose (the origin)
            #  This can be found with a closed form for the intersection of a segment and a circle
            idx_prev = idx_goal - 1
            px, py = self.path[idx_prev][0], self.path[idx_prev][1]
            gx, gy = self.path[idx_goal][0], self.path[idx_goal][1]

            # transform to the robot frame so that the circle centers at (0,0)
            prev_p = (px - self.robot.px, py - self.robot.py)
            goal_p = (gx - self.robot.px, gy - self.robot.py)
            i_points = MathHelper.circleSegmentIntersection(prev_p, goal_p, self.lookahead_dist)
            if len(i_points) == 0:
                # If there is no intersection, take the closest intersection point (foot of a perpendicular)
                # between the current position and the line segment
                i_points.append(MathHelper.closestPointOnLine(prev_p, goal_p))
            pt_x = i_points[0][0] + self.robot.px
            pt_y = i_points[0][1] + self.robot.py

        # calculate the angle on trajectory
        theta = self.angle(self.path[idx_prev], self.path[idx_goal])

        # calculate the curvature on trajectory
        if idx_goal == 1:
            idx_goal = idx_goal + 1
        idx_prev = idx_goal - 1
        idx_pprev = idx_prev - 1
        a = self.dist(self.path[idx_prev], self.path[idx_goal])
        b = self.dist(self.path[idx_pprev], self.path[idx_goal])
        c = self.dist(self.path[idx_pprev], self.path[idx_prev])
        cosB = (a * a + c * c - b * b) / (2 * a * c)
        sinB = math.sin(math.acos(cosB))
        cross = (self.path[idx_prev][0] - self.path[idx_pprev][0]) * \
                (self.path[idx_goal][1] - self.path[idx_pprev][1]) - \
                (self.path[idx_prev][1] - self.path[idx_pprev][1]) * \
                (self.path[idx_goal][0] - self.path[idx_pprev][0])
        kappa = math.copysign(2 * sinB / b, cross)

        return (pt_x, pt_y), theta, kappa

    def linearRegularization(self, v_d: float) -> float:
        """
        Linear velocity regularization

        Parameters:
            v_d (float): reference velocity input

        Returns:
            v (float): control velocity output
        """
        v_inc = v_d - self.robot.v
        v_inc = MathHelper.clamp(v_inc, self.params["MIN_V_INC"], self.params["MAX_V_INC"])

        v = self.robot.v + v_inc
        v = MathHelper.clamp(v, self.params["MIN_V"], self.params["MAX_V"])

        return v

    def angularRegularization(self, w_d: float) -> float:
        """
        Angular velocity regularization

        Parameters:
            w_d (float): reference angular velocity input

        Returns:
            w (float): control angular velocity output
        """
        w_inc = w_d - self.robot.w
        w_inc = MathHelper.clamp(w_inc, self.params["MIN_W_INC"], self.params["MAX_W_INC"])

        w = self.robot.w + w_inc
        w = MathHelper.clamp(w, self.params["MIN_W"], self.params["MAX_W"])

        return w

    def shouldMoveToGoal(self, cur: tuple, goal: tuple) -> bool:
        """
        Whether to move to the goal pose

        Parameters:
            cur (tuple): current pose of robot
            goal (tuple): goal pose of robot

        Returns:
            flag (bool): true if robot should perform movement
        """
        return self.dist(cur, goal) > self.params["GOAL_DIST_TOL"]
    
    def shouldRotateToPath(self, angle_to_path: float) -> bool:
        """
        Whether to correct the tracking path with rotation operation

        Parameters:
            angle_to_path (float): the angle deviation

        Returns:
            flag (bool): true if robot should perform rotation
        """
        return angle_to_path > self.params["ROTATE_TOL"]

    def reachGoal(self, cur: tuple, goal: tuple) -> bool:
        """
        Whether the robot has reached the goal pose

        Parameters:
            cur (tuple): current pose of robot
            goal (tuple): goal pose of robot

        Returns:
            flag (bool): true if robot has reached the goal
        """
        e_theta = self.regularizeAngle(cur[2] - goal[2])
        return not (self.shouldMoveToGoal((cur[0], cur[1]), (goal[0], goal[1]))
                    or self.shouldRotateToPath(abs(e_theta)))

    def isCollision(self, cur_pos: tuple):
        """
        Whether the robot is in collision with obstacles

        Parameters:
            cur_pos (tuple): current position of robot

        Returns:
            flag (bool): true if robot is in collision
        """
        obstacles = self.obstacles
        for obs in obstacles:
            if abs(cur_pos[0] - obs[0]) < 0.5 and abs(cur_pos[1] - obs[1]) < 0.5:
                return True
        return False