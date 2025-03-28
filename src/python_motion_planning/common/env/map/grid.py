"""
@file: grid.py
@breif: Grid Map for Path Planning
@author: Wu Maojia
@update: 2025.3.29
"""
from python_motion_planning.common.env.map import Map


class Grid(Map):
    """
    Class for Grid Map.

    Parameters:
        env: Base environment.

    Examples:
        >>> env = Env((30, 40))
        >>> map = Grid(env)
        >>> map
        Grid(Env(30, 40))
    """
    def __init__(self, env: Env) -> None:
        super().__init__(env)
    
    def __str__(self) -> str:
        return "Grid({})".format(self.env)

    def __repr__(self) -> str:
        return self.__str__()

    def getNeighbor(self, node: Node) -> list:
        """
        Get neighbor nodes of a given node.
        
        Parameters:
            node: Node to get neighbor nodes.
        
        Returns:
            nodes: List of neighbor nodes.
        """
        pass

    def inCollision(self, p1: PointND, p2: PointND) -> bool:
        """
        Check if the line of sight between two points is in collision.
        
        Parameters:
            p1: Start point of the line.
            p2: End point of the line.
        
        Returns:
            in_collision: True if the line of sight is in collision, False otherwise.
        """
        pass
        
    def getDistance(self, p1: PointND, p2: PointND) -> float:
        """
        Get the distance between two points.

        Parameters:
            p1: First point.
            p2: Second point.
        
        Returns:
            dist: Distance between two points.
        """
        pass