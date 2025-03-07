"""
@file: local_planner.py
@breif: Base class for local planner.
@author: Winter
@update: 2023.3.2
"""
import math

from python_motion_planning.common.structure import Grid
from python_motion_planning.common.math import MathHelper
from python_motion_planning.common.geometry import Point3d

class Controller:
    """
    Base class for local planner.

    Parameters:
        params (dict): parameters
    """
    def __init__(self, params: dict) -> None:
        # parameters
        self.params = params["strategy"]["controller"]
        # environment
        self.env = Grid(params)
        # obstacles
        self.obstacles = self.env.obstacles
        # robot
        self.robot = None

    @property
    def lookahead_dist(self):
        return MathHelper.clamp(
            abs(self.robot.v) * self.params["lookahead_time"],
            self.params["min_lookahead_dist"],
            self.params["max_lookahead_dist"]
        )

    def dist(self, start: Point3d, end: Point3d) -> float:
        return math.hypot(end.x() - start.x(), end.y() - start.y())
    
    def angle(self, start: Point3d, end: Point3d) -> float:
        return math.atan2(end.y() - start.y(), end.x() - start.x())

    def regularizeAngle(self, angle: float):
        return angle - 2.0 * math.pi * math.floor((angle + math.pi) / (2.0 * math.pi))

    def getLookaheadPoint(self, path: list):
        """
        Find the point on the path that is exactly the lookahead distance away from the robot

        Returns:
            lookahead_pt (tuple): lookahead point
            theta (float): the angle on trajectory
            kappa (float): the curvature on trajectory
        """
        if path is None:
            assert RuntimeError("Planner path is invalid!")

        # Find the first pose which is at a distance greater than the lookahead distance
        dist_to_robot = [self.dist(p, Point3d(self.robot.px, self.robot.py)) for p in path]
        idx_closest = dist_to_robot.index(min(dist_to_robot))
        idx_goal = len(path) - 1
        idx_prev = idx_goal - 1
        for i in range(idx_closest, len(path)):
            if self.dist(path[i], Point3d(self.robot.px, self.robot.py)) >= self.lookahead_dist:
                idx_goal = i
                break

        pt_x, pt_y = None, None
        if idx_goal == len(path) - 1:
            # If the no pose is not far enough, take the last pose
            pt_x = path[idx_goal][0]
            pt_y = path[idx_goal][1]
        else:
            if idx_goal == 0:
                idx_goal = idx_goal + 1
            #  find the point on the line segment between the two poses
            #  that is exactly the lookahead distance away from the robot pose (the origin)
            #  This can be found with a closed form for the intersection of a segment and a circle
            idx_prev = idx_goal - 1
            px, py = path[idx_prev][0], path[idx_prev][1]
            gx, gy = path[idx_goal][0], path[idx_goal][1]

            # transform to the robot frame so that the circle centers at (0,0)
            prev_p = (px - self.robot.px, py - self.robot.py)
            goal_p = (gx - self.robot.px, gy - self.robot.py)
            i_points = MathHelper.circleSegmentIntersection(prev_p, goal_p, self.lookahead_dist)
            if len(i_points) == 0:
                # If there is no intersection, take the closest intersection point (foot of a perpendicular)
                # between the current position and the line segment
                i_points.append(MathHelper.closestPointOnLine(prev_p, goal_p))
            else:
                dist_to_goal = float("inf")
                for i_point in i_points:
                    dist = math.hypot(i_point[0] + self.robot.px - gx, i_point[1] + self.robot.py - gy)
                    if dist < dist_to_goal:
                        dist_to_goal = dist
                        pt_x = i_point[0] + self.robot.px
                        pt_y = i_point[1] + self.robot.py

        # calculate the angle on trajectory
        theta = self.angle(path[idx_prev], path[idx_goal])

        # calculate the curvature on trajectory
        if idx_goal == 1:
            idx_goal = idx_goal + 1
        idx_prev = idx_goal - 1
        idx_pprev = idx_prev - 1
        a = self.dist(path[idx_prev],  path[idx_goal])
        b = self.dist(path[idx_pprev], path[idx_goal])
        c = self.dist(path[idx_pprev], path[idx_prev])
        cosB = MathHelper.clamp((a * a + c * c - b * b) / (2 * a * c), -1.0, 1.0)
        sinB = math.sin(math.acos(cosB))
        cross = (path[idx_prev][0] - path[idx_pprev][0]) * \
                (path[idx_goal][1] - path[idx_pprev][1]) - \
                (path[idx_prev][1] - path[idx_pprev][1]) * \
                (path[idx_goal][0] - path[idx_pprev][0])
        kappa = math.copysign(2 * sinB / b, cross)

        return Point3d(pt_x, pt_y, theta), kappa

    def linearRegularization(self, v_d: float) -> float:
        """
        Linear velocity regularization

        Parameters:
            v_d (float): reference velocity input

        Returns:
            v (float): control velocity output
        """
        v_inc = v_d - self.robot.v
        v_inc = MathHelper.clamp(v_inc, self.params["min_v_inc"], self.params["max_v_inc"])

        v = self.robot.v + v_inc
        v = MathHelper.clamp(v, self.params["min_v"], self.params["max_v"])

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
        w_inc = MathHelper.clamp(w_inc, self.params["min_w_inc"], self.params["max_w_inc"])

        w = self.robot.w + w_inc
        w = MathHelper.clamp(w, self.params["min_w"], self.params["max_w"])

        return w

    def shouldRotateToGoal(self, cur: Point3d, goal: Point3d) -> bool:
        """
        Whether to reach the target pose through rotation operation

        Parameters:
            cur (tuple): current pose of robot
            goal (tuple): goal pose of robot

        Returns:
            flag (bool): true if robot should perform rotation
        """
        return self.dist(cur, goal) < self.params["goal_dist_tol"]
    
    def shouldRotateToPath(self, angle_to_path: float, tol: float=None) -> bool:
        """
        Whether to correct the tracking path with rotation operation

        Parameters:
            angle_to_path (float): the angle deviation
            tol (float): the angle deviation tolerence

        Returns:
            flag (bool): true if robot should perform rotation
        """
        return ((tol is not None) and (angle_to_path > tol)) or (angle_to_path > self.params["rotate_tol"])