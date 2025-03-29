"""
@file: node.py
@breif: map node data stucture
@author: Wu Maojia, Yang Haodong 
@update: 2025.3.29
"""
from python_motion_planning.common.geometry.point import *

class Node(object):
    """
    Class for map nodes.

    Parameters:
        current: current point
        parent: point of parent node
        g: path cost
        h: heuristic cost

    Examples:
        >>> node1 = Node(Point2D(1, 0), Point2D(2, 3), 1, 2)
        >>> node2 = Node(Point2D(1, 0), Point2D(2, 3), 1, 2)
        >>> node3 = Node(Point2D(1, 1), Point2D(2, 3), 1, 2)

        >>> node1
        Node(Point2D([1.0, 0.0]), Point2D([2.0, 3.0]), 1, 2)
        
        >>> node1 + node2
        Node(Point2D([2.0, 0.0]), Point2D([2.0, 3.0]), 2, 2)

        >>> node1 == node2
        True
        
        >>> node1 != node3
        True

        >>> node1.current
        Point2D([1.0, 0.0])

        >>> node1.parent
        Point2D([2.0, 3.0])

        >>> node1.g
        1

        >>> node1.h
        2

        >>> node1.ndim
        2
    """
    def __init__(self, current: PointND, parent: PointND = None, g: float = 0, h: float = 0) -> None:
        self._current = current
        self._parent = parent
        self._g = g
        self._h = h

        if self.parent is not None and self.current.ndim != self.parent.ndim:
            raise ValueError("The dimension of current and parent must be the same.")
    
    def __add__(self, node: "Node") -> "Node":
        return Node(self._current + node._current, self._parent, self._g + node._g, self._h)

    def __eq__(self, node: "Node") -> bool:
        if not isinstance(node, Node):
            return False
        return self._current == node._current
    
    def __ne__(self, node: "Node") -> bool:
        return not self.__eq__(node)

    def __lt__(self, node: "Node") -> bool:
        assert isinstance(node, Node)
        return self._g + self._h < node._g + node._h or \
                (self._g + self._h == node._g + node._h and self._h < node._h)

    def __hash__(self) -> int:
        return hash(self._current)

    def __str__(self) -> str:
        return "Node({}, {}, {}, {})".format(self._current, self._parent, self._g, self._h)

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def current(self) -> PointND:
        return self._current

    @property
    def parent(self) -> PointND:
        return self._parent

    @property
    def g(self) -> float:
        return self._g

    @g.setter
    def g(self, value: float) -> None:
        self._g = value

    @property
    def h(self) -> float:
        return self._h

    @h.setter
    def h(self, value: float) -> None:
        self._h = value

    @property
    def ndim(self) -> int:
        return self.current.ndim