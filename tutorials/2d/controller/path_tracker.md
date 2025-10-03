Transform the planned path from map frame to world frame.

```python
path_world = map_.path_map_to_world(path)
print(path_world)
```

Print results:

```
[(5.5, 5.5), (5.5, 6.5), (5.5, 7.5), (5.5, 8.5), (5.5, 9.5), (5.5, 10.5), (5.5, 11.5), (5.5, 12.5), (6.5, 13.5), (6.5, 14.5), (6.5, 15.5), (7.5, 16.5), (7.5, 17.5), (8.5, 18.5), (9.5, 18.5), (10.5, 19.5), (11.5, 19.5), (12.5, 19.5), (13.5, 19.5), (14.5, 19.5), (15.5, 19.5), (16.5, 19.5), (17.5, 19.5), (18.5, 19.5), (19.5, 19.5), (20.5, 19.5), (21.5, 19.5), (22.5, 18.5), (23.5, 17.5), (24.5, 16.5), (25.5, 15.5), (26.5, 14.5), (27.5, 13.5), (28.5, 12.5), (29.5, 11.5), (30.5, 11.5), (31.5, 12.5), (32.5, 12.5), (33.5, 13.5), (34.5, 14.5), (35.5, 15.5), (36.5, 16.5), (37.5, 17.5), (38.5, 18.5), (39.5, 19.5), (40.5, 20.5), (41.5, 21.5), (42.5, 22.5), (43.5, 23.5), (44.5, 24.5), (45.5, 25.5)]
```

Create the toy simulator.

```python
dim = 2
env = ToySimulator(dim=dim, obstacle_grid=map_, robot_collisions=False)
```

Add robots.

```python
robots = {
    "1": CircularRobot(dim=dim, radius=1, pose=np.array([5.5, 5.5, 0]), vel=np.zeros(3),
                action_min=np.array([-2, -2, -3.14]), action_max=np.array([2, 2, 3.14]), color="C0", text="1"),
    "2": DiffDriveRobot(dim=dim, radius=1, pose=np.array([5.5, 5.5, 0]), vel=np.zeros(3),
                action_min=np.array([-2.82, 0, -6.28]), action_max=np.array([2.82, 0, 6.28]), color="C1", text="2")
}
```

Add controllers.

```python
controllers = {}
for rid, robot in robots.items():
    obs_space, act_space = env.build_robot_spaces(robot)
    controllers[rid] = PurePursuit(obs_space, act_space, env.dt, path_world, max_lin_speed=3, max_ang_speed=3.14)
    env.add_robot(rid, robot)
```

Simulate and render.

The visualizer has many customizable parameters. You can set them as you want. For example, if you want to show esdf map, set `show_esdf` to `True`. Here we set it to `False`.

```python
obs, _ = env.reset()

vis = Visualizer("Path Visualizer")
vis.render_toy_simulator(env, controllers, steps=300, show_traj=True, show_env_info=True, grid_kwargs={"show_esdf": False})
vis.plot_path(path, style="--", color="C4")
vis.show()
```

![pure_pursuit_2d.gif](../../../assets/pure_pursuit_2d.gif)

Print trajectory summary information.

```python
for rid in robots:
    ctrl = controllers[rid]
    print(rid, ":", vis.get_traj_info(rid, ctrl.goal, ctrl.goal_dist_tol, ctrl.goal_orient_tol))
vis.close()
```

Print results:

```
1 : {'traj_length': 64.05713763788278, 'success': True, 'dist_success': True, 'oracle_success': True, 'oracle_dist_success': True, 'success_time': 23.3, 'dist_success_time': 23.3, 'oracle_success_time': 20.8, 'oracle_dist_success_time': 20.8}
2 : {'traj_length': 61.7926006243001, 'success': True, 'dist_success': True, 'oracle_success': True, 'oracle_dist_success': True, 'success_time': 22.0, 'dist_success_time': 22.0, 'oracle_success_time': 20.400000000000002, 'oracle_dist_success_time': 20.400000000000002}
```

Runnable complete code:

```python
import random
random.seed(0)

import numpy as np
np.random.seed(0)

from python_motion_planning.common import *
from python_motion_planning.path_planner import *
from python_motion_planning.controller import *

map_ = Grid(bounds=[[0, 51], [0, 31]])

map_.fill_boundary_with_obstacles()
map_.type_map[10:21, 15] = TYPES.OBSTACLE
map_.type_map[20, :15] = TYPES.OBSTACLE
map_.type_map[30, 15:] = TYPES.OBSTACLE
map_.type_map[40, :16] = TYPES.OBSTACLE

map_.inflate_obstacles(radius=3)

start = (5, 5)
goal = (45, 25)

map_.type_map[start] = TYPES.START
map_.type_map[goal] = TYPES.GOAL

planner = AStar(map_=map_, start=start, goal=goal)
path, path_info = planner.plan()
print(path)
print(path_info)
map_.fill_expands(path_info["expand"])  # for visualizing the expanded nodes

path_world = map_.path_map_to_world(path)
print(path_world)

dim = 2
env = ToySimulator(dim=dim, obstacle_grid=map_, robot_collisions=False)

robots = {
    "1": CircularRobot(dim=dim, radius=1, pose=np.array([5.5, 5.5, 0]), vel=np.zeros(3),
                action_min=np.array([-2, -2, -3.14]), action_max=np.array([2, 2, 3.14]), color="C0", text="1"),
    "2": DiffDriveRobot(dim=dim, radius=1, pose=np.array([5.5, 5.5, 0]), vel=np.zeros(3),
                action_min=np.array([-2.82, 0, -6.28]), action_max=np.array([2.82, 0, 6.28]), color="C1", text="2")
}

controllers = {}
for rid, robot in robots.items():
    obs_space, act_space = env.build_robot_spaces(robot)
    controllers[rid] = PurePursuit(obs_space, act_space, env.dt, path_world, max_lin_speed=3, max_ang_speed=3.14)
    env.add_robot(rid, robot)

obs, _ = env.reset()

vis = Visualizer("Path Visualizer")
vis.render_toy_simulator(env, controllers, steps=300, show_traj=True, show_env_info=True, grid_kwargs={"show_esdf": False})
vis.plot_path(path, style="--", color="C4")
vis.show()

for rid in robots:
    ctrl = controllers[rid]
    print(rid, ":", vis.get_traj_info(rid, ctrl.goal, ctrl.goal_dist_tol, ctrl.goal_orient_tol))
vis.close()
```