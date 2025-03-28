"""
@file: env.py
@breif: 2-dimension environment
@author: Winter
@update: 2023.1.13
"""
from abc import ABC, abstractmethod

class Env(ABC):
    """
    Class for Motion Planning Base Environment. It is continuous and in Cartesian coordinate system

    Parameters:
        size: size of environment (length of size means the number of dimensions)


    Examples:
        >>> env = Env((30, 40))
    """
    def __init__(self, size: Iterable) -> None:
        try:
            self._size = tuple(size)
            if len(self._size) <= 1:
                raise ValueError("Input length must be greater than 1.")
            if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in self._size)
                raise ValueError("Input must be a non-empty 1D array.")
        except Exception as e:
            raise ValueError("Invalid input for Env: {}".format(e))

    def __str__(self) -> str:
        return "Env({})".format(self._size)

    def __repr__(self) -> str:
        return self.__str__()


class Grid2D(Env):
    """
    Class for discrete 2-d grid map.
    """
    def __init__(self, x_range: int, y_range: int) -> None:
        super().__init__(x_range, y_range)
        self.resolution = 0.5

        # # allowed motions
        # self.motions = [Node((-1, 0), None, 1, None), Node((-1, 1),  None, sqrt(2), None),
        #                 Node((0, 1),  None, 1, None), Node((1, 1),   None, sqrt(2), None),
        #                 Node((1, 0),  None, 1, None), Node((1, -1),  None, sqrt(2), None),
        #                 Node((0, -1), None, 1, None), Node((-1, -1), None, sqrt(2), None)]
        # obstacles
        self.obstacles = None
        self.obstacles_tree = None
        self.init()
    
    def init(self) -> None:
        """
        Initialize grid map.
        """
        x, y = self.x_range, self.y_range
        self.obstacles = np.zeros((x, y))
        self.obstacles[1,3] = 1
        self.obstacles[3,4] = 1
        self.obstacles[6,2] = 1

        # self.obstacles = obstacles
        # self.obstacles_tree = cKDTree(np.array(list(obstacles)))

    # def update(self, obstacles):
    #     self.obstacles = obstacles 
        # self.obstacles_tree = cKDTree(np.array(list(obstacles)))

    def getNeighbor(self, node):

    def getNodes(self):
        nodes = []
        # 把numpy数组为1的坐标里转换成(x,y)
        for i in range(self.x_range):
            for j in range(self.y_range):
                if self.obstacles[i, j] == 1:
                    nodes.append((i, j))
        return nodes


# class Map(Env):
#     """
#     Class for continuous 2-d map.
#     """
#     def __init__(self, x_range: int, y_range: int) -> None:
#         super().__init__(x_range, y_range)
#         self.boundary = None
#         self.obs_circ = None
#         self.obs_rect = None
#         self.init()

#     def init(self):
#         """
#         Initialize map.
#         """
#         x, y = self.x_range, self.y_range

#         # boundary of environment
#         self.boundary = [
#             [0, 0, 1, y],
#             [0, y, x, 1],
#             [1, 0, x, 1],
#             [x, 1, 1, y]
#         ]

#         # user-defined obstacles
#         self.obs_rect = [
#             [14, 12, 8, 2],
#             [18, 22, 8, 3],
#             [26, 7, 2, 12],
#             [32, 14, 10, 2]
#         ]

#         self.obs_circ = [
#             [7, 12, 3],
#             [46, 20, 2],
#             [15, 5, 2],
#             [37, 7, 3],
#             [37, 23, 3]
#         ]

#     def update(self, boundary, obs_circ, obs_rect):
#         self.boundary = boundary if boundary else self.boundary
#         self.obs_circ = obs_circ if obs_circ else self.obs_circ
#         self.obs_rect = obs_rect if obs_rect else self.obs_rect
