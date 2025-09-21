from typing import List, Tuple
import numpy as np

from .base_controller import BaseController

class PurePursuit(BaseController):
    """
    Pure Pursuit 路径跟踪控制器
    - path: List of waypoints (tuple or np.array)
    - lookahead_distance: 前瞻距离
    - max_acc: 最大加速度（限制控制器输出）
    """
    def __init__(self,
                 observation_space,
                 action_space,
                 path: List[Tuple[float, ...]],
                 lookahead_distance: float = 2.0):
        super().__init__(observation_space, action_space, path)
        self.lookahead_distance = lookahead_distance
        self.current_target_index = 0  # 当前路径索引

    def reset(self):
        self.current_target_index = 0

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        obs: [pos, vel, ...] 维度 >= 2*dim
        返回加速度向量 (ndarray, dim,)
        """
        if self.goal is None:
            return np.zeros(self.action_space.shape), self.goal

        dim = self.action_space.shape[0]
        pos = obs[:dim]
        vel = obs[dim:2*dim]

        # 1. 找到前瞻点
        target = self._get_lookahead_point(pos)

        # 2. 计算速度方向
        direction = target - pos
        distance = np.linalg.norm(direction)
        if distance > 1e-6:
            direction /= distance  # 单位向量

        # 3. 简单加速度策略: a = k*(desired_vel - current_vel)
        desired_speed = self.action_space.high  # 这里可以改成路径速度规划
        desired_vel = direction * desired_speed
        acc = desired_vel - vel
        acc = np.clip(acc, self.action_space.low, self.action_space.high)

        return acc, target

    def _get_lookahead_point(self, pos: np.ndarray) -> np.ndarray:
        """
        在路径折线上寻找与当前位置圆 (半径=lookahead_distance) 的交点。
        如果终点在圆内，直接选择终点。
        如果有多个交点，取沿路径向前的点。
        如果没有交点，取路径上与当前位置最接近的点。
        """
        # 将路径点转换为numpy数组便于计算
        path = np.array(self.path)
        lookahead_sq = self.lookahead_distance **2
        end_point = path[-1]  # 获取路径终点
        
        # 检查终点是否在当前位置的前瞻圆内
        end_dist_sq = np.dot(end_point - pos, end_point - pos)
        if end_dist_sq <= lookahead_sq + 1e-6:  # 加小量防止浮点误差
            return end_point
        
        candidates = []
        
        # 遍历路径中的每一段线段
        for i in range(len(path) - 1):
            # 当前线段的起点和终点
            p1 = path[i]
            p2 = path[i + 1]
            
            # 向量计算
            d = p2 - p1  # 线段方向向量
            v = pos - p1  # 从线段起点到当前位置的向量
            
            # 计算投影长度（线段参数t的分子）
            t_numerator = np.dot(v, d)
            t_denominator = np.dot(d, d)
            
            # 如果线段长度为0，跳过
            if t_denominator < 1e-10:
                continue
                
            t = t_numerator / t_denominator  # 线段参数t
            
            # 计算线段上最近点
            if t < 0.0:
                closest = p1
            elif t > 1.0:
                closest = p2
            else:
                closest = p1 + t * d
                
            # 计算最近点到当前位置的距离平方
            dist_sq = np.dot(pos - closest, pos - closest)
            
            # 如果距离小于等于前瞻距离，检查是否有交点
            if dist_sq <= lookahead_sq + 1e-6:  # 加小量防止浮点误差
                # 计算交点
                if t_denominator < 1e-10:
                    continue
                    
                # 求解二次方程得到交点参数
                a = t_denominator
                b = -2 * t_numerator
                c = np.dot(v, v) - lookahead_sq
                discriminant = b**2 - 4*a*c
                
                if discriminant < 0:
                    continue  # 无实根
                
                sqrt_d = np.sqrt(discriminant)
                t1 = (-b + sqrt_d) / (2*a)
                t2 = (-b - sqrt_d) / (2*a)
                
                # 检查交点是否在线段上
                for t_intersect in [t1, t2]:
                    if 0.0 <= t_intersect <= 1.0:
                        intersect_point = p1 + t_intersect * d
                        # 记录交点及其在路径中的位置
                        candidates.append((i + t_intersect, intersect_point))
        
        # 如果找到交点，选择路径中最靠后的那个
        if candidates:
            # 按路径位置排序，取最大的那个
            candidates.sort(key=lambda x: x[0])
            return candidates[-1][1]
        
        # 如果没有找到交点，返回路径上最近的点
        min_dist_sq = float('inf')
        closest_point = path[0]
        
        for point in path:
            dist_sq = np.dot(pos - point, pos - point)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_point = point
                
        return closest_point
