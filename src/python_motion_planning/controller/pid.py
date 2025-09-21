"""
@file: pid.py
@breif: PID motion planning
@author: Wu Maojia, Yang Haodong
@update: 2025.9.21
"""
import numpy as np
from typing import List, Tuple

from .pure_pursuit import PurePursuit

class PID(PurePursuit):
    """
    继承自PurePursuit的N维PID控制器
    支持Gymnasium接口，适用于任意维度的路径跟踪控制
    """
    def __init__(self,
                 observation_space,
                 action_space,
                 path: List[Tuple[float, ...]],
                 lookahead_distance: float = 2.0,
                 k_p: float = 1.0,
                 k_i: float = 0.1,
                 k_d: float = 0.1,
                 time_step: float = 0.1):
        # 调用父类构造函数
        super().__init__(observation_space, action_space, path, lookahead_distance)
        
        # PID参数 - 支持标量或向量形式（为不同维度设置不同参数）
        dim = action_space.shape[0]
        self.k_p = np.full(dim, k_p) if np.isscalar(k_p) else np.array(k_p)
        self.k_i = np.full(dim, k_i) if np.isscalar(k_i) else np.array(k_i)
        self.k_d = np.full(dim, k_d) if np.isscalar(k_d) else np.array(k_d)
        
        # 时间步长
        self.time_step = time_step
        
        # 初始化误差项
        self.reset()

    def reset(self):
        """重置控制器状态，继承自基类并添加PID特有状态"""
        super().reset()  # 调用父类的重置方法
        dim = self.action_space.shape[0]
        self.integral_error = np.zeros(dim)  # 积分误差
        self.prev_error = np.zeros(dim)      # 上一次的误差

    def get_action(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算PID控制动作
        obs: 观测值，格式为[pos, vel, ...]，前2*dim元素为位置和速度
        返回: (控制动作, 目标点)
        """
        if self.goal is None:
            return np.zeros(self.action_space.shape), self.goal
            
        dim = self.action_space.shape[0]
        pos = obs[:dim]       # 当前位置
        vel = obs[dim:2*dim]  # 当前速度
        
        # 1. 获取前瞻点（使用父类PurePursuit的方法）
        target = self._get_lookahead_point(pos)
        
        # 2. 计算期望速度（沿目标方向）
        direction = target - pos
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:
            # 到达目标点，速度设为0
            desired_vel = np.zeros(dim)
        else:
            # 沿方向向量的单位向量乘以最大速度
            desired_vel = (direction / distance) * self.action_space.high
        
        # 3. 计算速度误差
        error = desired_vel - vel
        
        # 4. 计算积分项（累加误差）
        self.integral_error += error * self.time_step
        
        # 5. 计算微分项（误差变化率）
        derivative_error = (error - self.prev_error) / self.time_step
        self.prev_error = error.copy()
        
        # 6. PID控制律计算加速度
        acc = (
            self.k_p * error + 
            self.k_i * self.integral_error + 
            self.k_d * derivative_error
        )
        
        # 7. 限制加速度在动作空间范围内
        acc = np.clip(acc, self.action_space.low, self.action_space.high)
        
        return acc, target


# class PID():
#     """
#     Class for PID motion planning.

#     Parameters:
#         start (tuple): start point coordinate
#         goal (tuple): goal point coordinate
#         env (Env): environment
#         heuristic_type (str): heuristic function type
#         **params: other parameters can be found in the parent class LocalPlanner

#     Examples:
#         >>> from python_motion_planning.utils import Grid
#         >>> from python_motion_planning.local_planner import PID
#         >>> start = (5, 5, 0)
#         >>> goal = (45, 25, 0)
#         >>> env = Grid(51, 31)
#         >>> planner = PID(start, goal, env)
#         >>> planner.run()
#     """
#     def __init__(self, start: tuple, goal: tuple, env: Env, heuristic_type: str = "euclidean",
#                  k_v_p: float = 1.00, k_v_i: float = 0.10, k_v_d: float = 0.10,
#                  k_w_p: float = 1.00, k_w_i: float = 0.10, k_w_d: float = 0.10,
#                  k_theta: float = 0.75, **params) -> None:
#         super().__init__(start, goal, env, heuristic_type, MIN_LOOKAHEAD_DIST=0.75, **params)
#         # PID parameters
#         self.e_w, self.i_w = 0.0, 0.0
#         self.e_v, self.i_v = 0.0, 0.0
#         self.k_v_p, self.k_v_i, self.k_v_d = k_v_p, k_v_i, k_v_d
#         self.k_w_p, self.k_w_i, self.k_w_d = k_w_p, k_w_i, k_w_d
#         self.k_theta = k_theta

#         # global planner
#         g_start = (start[0], start[1])
#         g_goal  = (goal[0], goal[1])
#         self.g_planner = {"planner_name": "a_star", "start": g_start, "goal": g_goal, "env": env}
#         self.path = self.g_path[::-1]

#     def __str__(self) -> str:
#         return "PID Planner"

#     def plan(self):
#         """
#         PID motion plan function.

#         Returns:
#             flag (bool): planning successful if true else failed
#             pose_list (list): history poses of robot
#         """
#         dt = self.params["TIME_STEP"]
#         for _ in range(self.params["MAX_ITERATION"]):
#             # break until goal reached
#             if self.reachGoal(tuple(self.robot.state.squeeze(axis=1)[0:3]), self.goal):
#                 return True, self.robot.history_pose
            
#             # find next tracking point
#             lookahead_pt, theta_trj, _ = self.getLookaheadPoint()

#             # desired angle
#             theta_err = self.angle(self.robot.position, lookahead_pt)
#             if abs(theta_err - theta_trj) > np.pi:
#                 if theta_err > theta_trj:
#                     theta_trj += 2 * np.pi
#                 else:
#                     theta_err += 2 * np.pi
#             theta_d = self.k_theta * theta_err + (1 - self.k_theta) * theta_trj
    
#             # calculate velocity command
#             e_theta = self.regularizeAngle(self.robot.theta - self.goal[2])
#             if not self.shouldMoveToGoal(self.robot.position, self.goal):
#                 if not self.shouldRotateToPath(abs(e_theta)):
#                     u = np.array([[0], [0]])
#                 else:
#                     u = np.array([[0], [self.angularRegularization(e_theta / dt)]])
#             else:
#                 e_theta = self.regularizeAngle(theta_d - self.robot.theta)
#                 if self.shouldRotateToPath(abs(e_theta)):
#                     u = np.array([[0], [self.angularRegularization(e_theta / dt)]])
#                 else:
#                     v_d = self.dist(lookahead_pt, self.robot.position) / dt
#                     u = np.array([[self.linearRegularization(v_d)], [self.angularRegularization(e_theta / dt)]])

#             # feed into robotic kinematic
#             self.robot.kinematic(u, dt)
        
#         return False, None

#     def run(self):
#         """
#         Running both plannig and animation.
#         """
#         _, history_pose = self.plan()
#         if not history_pose:
#             raise ValueError("Path not found and planning failed!")

#         path = np.array(history_pose)[:, 0:2]
#         cost = np.sum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1, keepdims=True)))
#         self.plot.plotPath(self.path, path_color="r", path_style="--")
#         self.plot.animation(path, str(self), cost, history_pose=history_pose)

#     def linearRegularization(self, v_d: float) -> float:
#         """
#         Linear velocity controller with pid.

#         Parameters:
#             v_d (float): reference velocity input

#         Returns:
#             v (float): control velocity output
#         """
#         e_v = v_d - self.robot.v
#         self.i_v += e_v * self.params["TIME_STEP"]
#         d_v = (e_v - self.e_v) / self.params["TIME_STEP"]
#         self.e_v = e_v

#         v_inc = self.k_v_p * e_v + self.k_v_i * self.i_v + self.k_v_d * d_v
#         v_inc = MathHelper.clamp(v_inc, self.params["MIN_V_INC"], self.params["MAX_V_INC"])

#         v = self.robot.v + v_inc
#         v = MathHelper.clamp(v, self.params["MIN_V"], self.params["MAX_V"])
        
#         return v

#     def angularRegularization(self, w_d: float) -> float:
#         """
#         Angular velocity controller with pid.

#         Parameters:
#             w_d (float): reference angular input

#         Returns:
#             w (float): control angular velocity output
#         """
#         e_w = w_d - self.robot.w
#         self.i_w += e_w * self.params["TIME_STEP"]
#         d_w = (e_w - self.e_w) / self.params["TIME_STEP"]
#         self.e_w = e_w

#         w_inc = self.k_w_p * e_w + self.k_w_i * self.i_w + self.k_w_d * d_w
#         w_inc = MathHelper.clamp(w_inc, self.params["MIN_W_INC"], self.params["MAX_W_INC"])

#         w = self.robot.w + w_inc
#         w = MathHelper.clamp(w, self.params["MIN_W"], self.params["MAX_W"])

#         return w