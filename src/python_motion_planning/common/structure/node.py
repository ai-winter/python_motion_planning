"""
@file: node.py
@breif: 2-dimension node data stucture
@author: Winter
@update: 2023.1.13
"""
from python_motion_planning.common.geometry import Point3d

class Node(object):
    """
    Class for searching nodes.

    Parameters:
        current (Point3d): current coordinate
        parent (Point3d): coordinate of parent node
        g (float): path cost
        h (float): heuristic cost
    """
    def __init__(self, current: Point3d, parent: Point3d=None, g: float=0, h: float=0) -> None:
        self.current = current
        self.parent = parent
        self.g = g
        self.h = h
    
    def __add__(self, node: 'Node') -> 'Node':
        if not isinstance(node, type(self)):
            raise NotImplementedError
        return Node(
            Point3d(self.current.x() + node.current.x(),
                    self.current.y() + node.current.y(),
                    self.current.theta() + node.current.theta(),
            ), self.parent, self.g + node.g, self.h
        )

    def __eq__(self, node: 'Node') -> bool:
        if not isinstance(node, type(self)):
            raise NotImplementedError
        return (self.x == node.x and self.y == node.y)
    
    def __ne__(self, node: 'Node') -> bool:
        return not self.__eq__(node)

    def __lt__(self, node: 'Node') -> bool:
        if not isinstance(node, type(self)):
            raise NotImplementedError
        return self.g + self.h < node.g + node.h or \
                (self.g + self.h == node.g + node.h and self.h < node.h)

    def __hash__(self) -> int:
        return hash(self.current)

    def __str__(self) -> str:
        return f"----------\ncurrent:({self.x}, {self.y})\nparent:({self.px}, {self.py})\ng:{self.g}\nh:{self.h}\n----------"
    
    @property
    def x(self) -> float:
        return self.current.x()
    
    @property
    def y(self) -> float:
        return self.current.y()

    @property
    def px(self) -> float:
        if self.parent is not None:
            return self.parent.x()
        else:
            return None

    @property
    def py(self) -> float:
        if self.parent is not None:
            return self.parent.y()
        else:
            return None