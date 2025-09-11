import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def create_9x9x9_cube_with_voxels():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a 9x9x9 boolean array
    filled = np.zeros((9, 9, 9), dtype=bool)
    

    filled[3:6, 3:6, 3:6] = True
    
    # Add some random cubes
    np.random.seed(42)
    for _ in range(50):
        x, y, z = np.random.randint(0, 9, 3)
        filled[x, y, z] = True
    
    # Create colors for the cubes
    colors = np.empty(filled.shape + (4,))  # RGBA
    colors[:, :, :, 3] = 0.8  # Alpha
    
    # Color different regions differently
    colors[filled] = [0.2, 0.8, 0.2, 0.8]  # Green cubes
    colors[3:6, 3:6, 3:6] = [0.8, 0.2, 0.2, 0.8]  # Red center cube
    
    ax.voxels(filled, facecolors=colors, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('9x9x9 Cube Grid using voxels()')
    
    plt.show()

# Method 2: Using scatter with cube markers (approximation)
def create_9x9x9_cube_with_scatter():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate all possible positions in a 9x9x9 grid
    x_pos, y_pos, z_pos = [], [], []
    
    # Create a hollow cube shell
    for x in range(9):
        for y in range(9):
            for z in range(9):
                # Only include cubes on the edges (hollow cube)
                if (x == 0 or x == 8 or 
                    y == 0 or y == 8 or 
                    z == 0 or z == 8):
                    x_pos.append(x)
                    y_pos.append(y)
                    z_pos.append(z)
    
    # Use square markers with large size to approximate cubes
    ax.scatter(x_pos, y_pos, z_pos, marker='s', s=200, 
              c='blue', alpha=0.7, edgecolors='black')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('9x9x9 Cube Grid using scatter()')
    
    # Set equal aspect ratio
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_zlim(0, 8)
    
    plt.show()

# Method 3: Custom function to create your specific pattern
def create_custom_cube_pattern():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a 9x9x9 boolean array
    filled = np.zeros((9, 9, 9), dtype=bool)
    
    # custom logic here - example patterns:
    
    # Pattern 1: Diagonal line
    for i in range(9):
        filled[i, i, i] = True
    
    # Pattern 2: Cross pattern in the middle layer
    filled[4, :, 4] = True  # Horizontal line
    filled[:, 4, 4] = True  # Vertical line
    
    # Pattern 3: Random cubes
    np.random.seed(19680801)
    for _ in range(100):
        x = np.random.randint(0, 9)
        y = np.random.randint(0, 9)
        z = np.random.randint(0, 9)
        filled[x, y, z] = True
    
    # Create different colors for different patterns
    colors = np.empty(filled.shape + (4,))
    
    # Default color for all cubes
    colors[filled] = [0.3, 0.7, 0.9, 0.8]  # Light blue
    
    # Special color for diagonal
    for i in range(9):
        if filled[i, i, i]:
            colors[i, i, i] = [0.9, 0.3, 0.3, 0.8]  # Red
    
    # Special color for cross pattern
    colors[4, :, 4] = [0.3, 0.9, 0.3, 0.8]  # Green
    colors[:, 4, 4] = [0.3, 0.9, 0.3, 0.8]  # Green
    
    ax.voxels(filled, facecolors=colors, edgecolors='black', linewidth=0.3)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Custom 9x9x9 Cube Pattern')
    
    plt.show()

def create_cubes_from_coordinates(coordinates):
    """
    Create cubes from a list of (x, y, z) coordinates
    
    Parameters:
    coordinates: list of tuples [(x1, y1, z1), (x2, y2, z2), ...]
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Find the bounds
    if not coordinates:
        return
    
    max_coord = max(max(coord) for coord in coordinates)
    size = max_coord + 1
    
    # Create boolean array
    filled = np.zeros((size, size, size), dtype=bool)
    
    # Fill the specified coordinates
    for x, y, z in coordinates:
        filled[x, y, z] = True
    
    # Create colors
    colors = np.empty(filled.shape + (4,))
    colors[filled] = [0.2, 0.8, 0.6, 0.8]  # Teal color
    
    ax.voxels(filled, facecolors=colors, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Cubes from {len(coordinates)} coordinates')
    
    plt.show()

if __name__ == "__main__":
    create_9x9x9_cube_with_voxels()
    create_custom_cube_pattern()
    example_coords = [(0,0,0), (1,1,1), (2,2,2), (0,1,2), (3,3,3), (4,4,4)]
    create_cubes_from_coordinates(example_coords)