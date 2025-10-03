Define start and goal points.

```python
start = (2, 2, 2)
goal = (18, 18, 18)
```

Add the start and goal points to the map.
```python
map_.type_map[start] = TYPES.START
map_.type_map[goal] = TYPES.GOAL
```

Create the path-planner and plan the path.
```python
planner = AStar(map_=map_, start=start, goal=goal)
path, path_info = planner.plan()
print(path)
print(path_info)
```

Print results:
```
[(2, 2, 2), (3, 2, 2), (4, 2, 3), (5, 2, 3), (6, 2, 4), (7, 2, 5), (8, 2, 5), (9, 2, 5), (10, 2, 6), (11, 3, 7), (12, 3, 8), (13, 3, 9), (13, 3, 10), (14, 4, 11), (14, 4, 12), (15, 5, 13), (16, 6, 14), (17, 7, 15), (18, 8, 16), (18, 9, 16), (18, 10, 16), (18, 11, 16), (18, 12, 16), (18, 13, 16), (18, 14, 16), (18, 15, 16), (18, 16, 17), (18, 17, 18), (18, 18, 18)]
{'success': True, 'start': (2, 2, 2), 'goal': (18, 18, 18), 'length': 35.70601334439802, 'cost': 35.70601334439802, 'expand': {(2, 2, 2): Node((2, 2, 2), None, 0, 27.712812921102035), ...}}
```

Visualize.
```python
vis = Visualizer("Path Visualizer")
vis.plot_grid_map(map_, equal=False)
vis.plot_path(path, style="-", color="C2")
vis.show()
vis.close()
```

![a_star_3d.svg](../../../assets/a_star_3d.svg)

Runnable complete code:

```python
import random
random.seed(0)

import numpy as np
np.random.seed(0)

from python_motion_planning.common import *
from python_motion_planning.path_planner import *
from python_motion_planning.controller import *

map_ = Grid(bounds=[[0, 21], [0, 21], [0, 21]], resolution=1.0)
map_.type_map[:, 7, 0:11] = TYPES.OBSTACLE
map_.type_map[6:11, 8:13, :] = TYPES.OBSTACLE
map_.type_map[14, 13:, 11:] = TYPES.OBSTACLE
map_.type_map[6:11, 0:8, 11] = TYPES.OBSTACLE
map_.inflate_obstacles(radius=3)

start = (2, 2, 2)
goal = (18, 18, 18)

map_.type_map[start] = TYPES.START
map_.type_map[goal] = TYPES.GOAL

planner = AStar(map_=map_, start=start, goal=goal)
path, path_info = planner.plan()
print(path)
print(path_info)

vis = Visualizer("Path Visualizer")
vis.plot_grid_map(map_)
vis.plot_path(path, style="-", color="C2")
vis.show()
vis.close()
```