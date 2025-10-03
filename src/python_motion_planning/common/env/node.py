"""
@file: node.py
@author: Wu Maojia, Yang Haodong 
@update: 2025.10.3
"""
from __future__ import annotations

import numpy as np


class Node(object):
    """
    Class for map nodes.

    Args:
        current: current point
        parent: point of parent node
        g: path cost
        h: heuristic cost

    Examples:
        >>> node1 = Node((1, 0), (2, 3), 1, 2)
        >>> node2 = Node((1, 0), (2, 3), 1, 2)
        >>> node3 = Node((1, 1), (2, 3), 1, 2)

        >>> node1
        Node((1, 0), (2, 3), 1, 2)
        
        >>> node1 + node2
        Node((2, 0), (1, 0), 2, 2)

        >>> node1 == node2
        True
        
        >>> node1 != node3
        True

        >>> node1.current
        (1, 0)

        >>> node1.parent
        (2, 3)

        >>> node1.g
        1

        >>> node1.h
        2

        >>> node1.dim
        2
    """
    def __init__(self, current: tuple, parent: tuple = None, g: float = 0, h: float = 0) -> None:
        self._current = current
        self.parent = parent
        self._g = g
        self._h = h

        if self.parent is not None and len(self.current) != len(self.parent):
            raise ValueError("The dimension of current " + str(self.current) + " and parent " + str(self.parent) + " must be the same.")
    
    def __add__(self, node: Node) -> Node:
        return Node(tuple(x+y for x, y in zip(self._current, node._current)), self._current, self._g + node._g, self._h)

    def __eq__(self, node: Node) -> bool:
        if not isinstance(node, Node):
            return False
        return self._current == node._current
    
    def __ne__(self, node: Node) -> bool:
        return not self.__eq__(node)

    def __lt__(self, node: Node) -> bool:
        return self._g + self._h < node._g + node._h or \
                (self._g + self._h == node._g + node._h and self._h < node._h)

    def __hash__(self) -> int:
        return hash(self._current)

    def __str__(self) -> str:
        return "Node({}, {}, {}, {})".format(self._current, self.parent, self._g, self._h)

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return len(self.current)

    @property
    def current(self) -> tuple:
        return self._current

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
    def dim(self) -> int:
        return len(self.current)