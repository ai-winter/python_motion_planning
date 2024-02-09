"""
@file: node.py
@breif: 2-dimension node data stucture
@author: Winter
@update: 2023.1.13
"""

class Node(object):
    """
    Class for searching nodes.

    Parameters:
        current (tuple): current coordinate
        parent (tuple): coordinate of parent node
        g (float): path cost
        h (float): heuristic cost

    Examples:
        >>> from env import Node
        >>> node1 = Node((1, 0), (2, 3), 1, 2)
        >>> node2 = Node((1, 0), (2, 5), 2, 8)
        >>> node3 = Node((2, 0), (1, 6), 3, 1)
        ...
        >>> node1 + node2
        >>> Node((2, 0), (2, 3), 3, 2)
        ...
        >>> node1 == node2
        >>> True
        ...
        >>> node1 != node3
        >>> True
    """
    def __init__(self, current: tuple, parent: tuple=None, g: float=0, h: float=0) -> None:
        self.current = current
        self.parent = parent
        self.g = g
        self.h = h
    
    def __add__(self, node):
        return Node((self.x + node.x, self.y + node.y), self.parent, self.g + node.g, self.h)

    def __eq__(self, node) -> bool:
        return self.current == node.current
    
    def __ne__(self, node) -> bool:
        return not self.__eq__(node)

    def __lt__(self, node) -> bool:
        return self.g + self.h < node.g + node.h or \
                (self.g + self.h == node.g + node.h and self.h < node.h)

    def __hash__(self) -> int:
        return hash(self.current)

    def __str__(self) -> str:
        return "----------\ncurrent:{}\nparent:{}\ng:{}\nh:{}\n----------" \
            .format(self.current, self.parent, self.g, self.h)
    
    @property
    def x(self) -> float:
        return self.current[0]
    
    @property
    def y(self) -> float:
        return self.current[1]

    @property
    def px(self) -> float:
        if self.parent:
            return self.parent[0]
        else:
            return None

    @property
    def py(self) -> float:
        if self.parent:
            return self.parent[1]
        else:
            return None