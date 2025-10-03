Fix the random seed to ensure reproducible results.

```python
import random
random.seed(0)

import numpy as np
np.random.seed(0)
```

Import necessary modules.

```python
from python_motion_planning.common import *
from python_motion_planning.path_planner import *
from python_motion_planning.controller import *
```

Define the grid map and add and inflate obstacles.

```python
map_ = Grid(bounds=[[0, 21], [0, 21], [0, 21]], resolution=1.0)
map_.type_map[:, 7, 0:11] = TYPES.OBSTACLE
map_.type_map[6:11, 8:13, :] = TYPES.OBSTACLE
map_.type_map[14, 13:, 11:] = TYPES.OBSTACLE
map_.type_map[6:11, 0:8, 11] = TYPES.OBSTACLE
map_.inflate_obstacles(radius=3)
```

Visualize to check the map.

```python
vis = Visualizer("Path Visualizer")
vis.plot_grid_map(map_)
vis.show()
vis.close()
```

![grid_map_3d.svg](../../../assets/grid_map_3d.svg)

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

vis = Visualizer("Path Visualizer")
vis.plot_grid_map(map_)
vis.show()
vis.close()
```