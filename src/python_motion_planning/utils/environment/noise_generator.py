import random
import numpy as np

def random_obstacles(x_range: int, y_range: int, z_range: int, amount_of_cubes: int):
    """
    Generates random obstacles within the matrix.

    Args:
        x (int): x-axis range of environment
        y (int): y-axis range of environment
        z (int): z-axis range of environment
        p (float): probability of generating an obstacle while iterating through the whole matrix

    Returns:
        obstacles (set): Set of obstacles to be added to the environment
    """
    if amount_of_cubes > x_range * y_range * z_range:
        raise ValueError(f"argument amount_of_cubes is larger than the available area in the matrix, max is: {x_range * y_range * z_range}")


    obstacles = set()

    for _ in range(amount_of_cubes):
        x = random.randint(0, x_range - 1) # -1 since it is 0-indexed in the actual obstacles
        y = random.randint(0, y_range - 1) # but the dimensions are 1-indexed
        z = random.randint(0, z_range - 1)

        obstacles.add((x, y, z))

    return obstacles