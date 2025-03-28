# """
# @file: collision.py
# @breif: Useful function relate to collision.
# @author: Winter
# @update: 2024.4.14
# """
# import numpy as np

# from python_motion_planning.common.env.types import TYPES
# from python_motion_planning.common.geometry.point import Point3d

# class CollisionChecker:
#     def __init__(self, grid_map: np.ndarray) -> None:
#         self.grid_map = grid_map

#     def __call__(self, node1: Point3d, node2: Point3d, 
#     method: str="bresenham") -> bool:
#         if method == "bresenham":
#             return self.bresenham(node1, node2)
#         elif method == "onestep":
#             return self.onestep(node1, node2)
#         else:
#             raise NotImplementedError

#     def update(self, grid_map: np.ndarray) -> None:
#         self.grid_map = grid_map

#     def onestep(self, node1: Point3d, node2: Point3d) -> bool:
#         """
#         Judge collision when moving one step from node1 to node2.

#         Parameters:
#             node1 (Point3d): node 1
#             node2 (Point3d): node 2

#         Returns:
#             collision (bool): True if collision exists else False
#         """
#         x1, y1 = round(node1.x()), round(node1.y())
#         x2, y2 = round(node2.x()), round(node2.y())

#         y_range, x_range = self.grid_map.shape
#         if x1 < 0 or x1 >= x_range or x2 < 0 or x2 >= x_range:
#             return True
#         if y1 < 0 or y1 >= y_range or y2 < 0 or y2 >= y_range:
#             return True

#         if self.grid_map[y1, x1] != TYPES.FREE_GRID or self.grid_map[y2, x2] != TYPES.FREE_GRID:
#             return True

#         if x1 != x2 and y1 != y2:
#             if x2 - x1 == y1 - y2:
#                 s1 = (min(x1, x2), min(y1, y2))
#                 s2 = (max(x1, x2), max(y1, y2))
#             else:
#                 s1 = (min(x1, x2), max(y1, y2))
#                 s2 = (max(x1, x2), min(y1, y2))
#             if self.grid_map[s1[1], s1[0]] != TYPES.FREE_GRID or self.grid_map[s2[1], s2[0]] != TYPES.FREE_GRID:
#                 return True
#         return False

#     def bresenham(self, node1: Point3d, node2: Point3d) -> bool:
#         """
#         Judge collision when moving from node1 to node2 using Bresenham.
#         Only useful in discrete grids

#         Parameters:
#             node1 (Point3d): start node
#             node2 (Point3d): end node

#         Returns:
#             collision (bool): True if collision occurs else False
#         """
#         x1, y1 = round(node1.x()), round(node1.y())
#         x2, y2 = round(node2.x()), round(node2.y())

#         y_range, x_range = self.grid_map.shape
#         if x1 < 0 or x1 >= x_range or x2 < 0 or x2 >= x_range:
#             return True
#         if y1 < 0 or y1 >= y_range or y2 < 0 or y2 >= y_range:
#             return True
            
#         if self.grid_map[y1, x1] != TYPES.FREE_GRID or self.grid_map[y2, x2] != TYPES.FREE_GRID:
#             return True

#         d_x = abs(x2 - x1)
#         d_y = abs(y2 - y1)
#         s_x = 0 if (x2 - x1) == 0 else (x2 - x1) / d_x
#         s_y = 0 if (y2 - y1) == 0 else (y2 - y1) / d_y
#         x, y, e = x1, y1, 0

#         # check if any obstacle exists between node1 and node2
#         if d_x > d_y:
#             tao = (d_y - d_x) / 2
#             while not x == x2:
#                 if e > tao:
#                     x = x + s_x
#                     e = e - d_y
#                 elif e < tao:
#                     y = y + s_y
#                     e = e + d_x
#                 else:
#                     x = x + s_x
#                     y = y + s_y
#                     e = e + d_x - d_y
#                 if self.grid_map[round(y), round(x)] != TYPES.FREE_GRID:
#                     return True
                    
#         # swap x and y
#         else:
#             tao = (d_x - d_y) / 2
#             while not y == y2:
#                 if e > tao:
#                     y = y + s_y
#                     e = e - d_x
#                 elif e < tao:
#                     x = x + s_x
#                     e = e + d_y
#                 else:
#                     x = x + s_x
#                     y = y + s_y
#                     e = e + d_y - d_x
#                 if self.grid_map[round(y), round(x)] != TYPES.FREE_GRID:
#                     return True
        
#         return False