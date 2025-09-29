
# Introduction

`Motion planning` plans the state sequence of the robot without conflict between the start and goal. 

`Motion planning` mainly includes `Path planning` and `Trajectory planning`.

* `Path Planning`: It's based on path constraints (such as obstacles), planning the optimal path sequence for the robot to travel without conflict between the start and goal.
* `Trajectory planning`: It plans the motion state to approach the global path based on kinematics, dynamics constraints and path sequence.

This repository provides the implementations of common `Motion planning` algorithms. **Your stars and forks are welcome**. Maintaining this repository requires a huge amount of work. **Therefore, you are also welcome to contribute to this repository by opening issues.**. If you want to submit pull requests or joining our development team, please [contact](#contact) us before you contribute.

The theory analysis can be found at [motion-planning](https://blog.csdn.net/frigidwinter/category_11410243.html).

We also provide [ROS C++](https://github.com/ai-winter/ros_motion_planning) version and [Matlab](https://github.com/ai-winter/matlab_motion_planning) version.

# Quick Start

## Overview
The source file structure is shown below

```
python_motion_planning
├─global_planner
|   ├─graph_search
|   ├─sample_search
|   └─evolutionary_search
├─local_planner
├─curve_generation
└─utils
    ├─agent
    ├─environment
    ├─helper
    ├─planner
    └─plot
```

* The global planning algorithm implementation is in the folder `global_planner` with `graph_search`, `sample_search` and `evolutionary search`.

* The local planning algorithm implementation is in the folder `local_planner`.

* The curve generation algorithm implementation is in the folder `curve_generation`.

## Install
*(Optional)* The code was tested in python=3.10. We recommend using `conda` to install the dependencies.

```shell
conda create -n pmp python=3.10
conda activate pmp
```

To install the repository, please run the following command in shell.

```shell
pip install python-motion-planning
```

## Run
Below are some simple examples.

1. Run planning and animation separately
```python
import python_motion_planning as pmp
planner = pmp.AStar(start=(5, 5), goal=(45, 25), env=pmp.Grid(51, 31))
cost, path, expand = planner.plan()
planner.plot.animation(path, str(planner), cost, expand)  # animation
```

2. Run planning and animation in one step
```python
import python_motion_planning as pmp
planner = pmp.AStar(start=(5, 5), goal=(45, 25), env=pmp.Grid(51, 31))
planner.run()       # run both planning and animation
```

3. Create planner in factory mode
```python
import python_motion_planning as pmp
search_factory = pmp.SearchFactory()
planner = search_factory("a_star", start=(5, 5), goal=(45, 25), env=pmp.Grid(51, 31))
planner.run()       # run both planning and animation
```

More examples can be found in the folder `examples` in the repository.

## Documentation

For more details, you can refer to [online documentation](https://ai-winter.github.io/python_motion_planning/).

The documentation is auto-generated using mkdocs. To do this, enter the root directory and run

```shell
python generate_mkdocs.py
mkdocs serve
```

Then open the browser and go to [http://127.0.0.1:8000](http://127.0.0.1:8000). That is the generated documentation.

# Version
## Global Planner

Planner      | Version                                                                                                                                                                         | Animation
------------ |---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| --------- 
**GBFS**              | [![Status](https://img.shields.io/badge/done-v1.0-brightgreen)](https://github.com/ai-winter/python_motion_planning/blob/master/global_planner/graph_search/gbfs.py)            | ![gbfs_python.png](assets/gbfs_python.png) 
**Dijkstra**                 | [![Status](https://img.shields.io/badge/done-v1.0-brightgreen)](https://github.com/ai-winter/python_motion_planning/blob/master/global_planner/graph_search/dijkstra.py)        | ![dijkstra_python.png](assets/dijkstra_python.png)
**A***               | [![Status](https://img.shields.io/badge/done-v1.0-brightgreen)](https://github.com/ai-winter/python_motion_planning/blob/master/global_planner/graph_search/a_star.py)          |  ![a_star_python.png](assets/a_star_python.png) 
**JPS**                 | [![Status](https://img.shields.io/badge/done-v1.0-brightgreen)](https://github.com/ai-winter/python_motion_planning/blob/master/global_planner/graph_search/jps.py)             | ![jps_python.png](assets/jps_python.png)
**D***                  | [![Status](https://img.shields.io/badge/done-v1.0-brightgreen)](https://github.com/ai-winter/python_motion_planning/blob/master/global_planner/graph_search/d_star.py)          | ![d_star_python.png](assets/d_star_python.png)
**LPA***                 | [![Status](https://img.shields.io/badge/done-v1.0-brightgreen)](https://github.com/ai-winter/python_motion_planning/blob/master/global_planner/graph_search/lpa_star.py)        | ![lpa_star_python.png](assets/lpa_star_python.png) 
**D\* Lite**                | [![Status](https://img.shields.io/badge/done-v1.0-brightgreen)](https://github.com/ai-winter/python_motion_planning/blob/master/global_planner/graph_search/d_star_lite.py)     | ![d_star_lite_python.png](assets/d_star_lite_python.png)
**Theta\***                | [![Status](https://img.shields.io/badge/done-v1.0-brightgreen)](https://github.com/ai-winter/python_motion_planning/blob/master/global_planner/graph_search/theta_star.py)      | ![theta_star_python.png](assets/theta_star_python.png)
**Lazy Theta\***                | [![Status](https://img.shields.io/badge/done-v1.0-brightgreen)](https://github.com/ai-winter/python_motion_planning/blob/master/global_planner/graph_search/lazy_theta_star.py) | ![lazy_theta_star_python.png](assets/lazy_theta_star_python.png)
**S-Theta\***                | [![Status](https://img.shields.io/badge/done-v1.0-brightgreen)](https://github.com/ai-winter/python_motion_planning/blob/master/global_planner/graph_search/s_theta_star.py)    | ![s_theta_star_python.png](assets/s_theta_star_python.png)
**Anya**                | [![Status](https://img.shields.io/badge/develop-v1.0-red)](https://github.com/ai-winter/python_motion_planning/blob/master/global_planner/graph_search/anya.py)                 | ![Status](https://img.shields.io/badge/gif-none-yellow)
**Voronoi**                | [![Status](https://img.shields.io/badge/done-v1.0-brightgreen)](https://github.com/ai-winter/python_motion_planning/blob/master/global_planner/graph_search/voronoi.py)         | ![voronoi_python.png](assets/voronoi_python.png) 
**RRT**                 | [![Status](https://img.shields.io/badge/done-v1.0-brightgreen)](https://github.com/ai-winter/python_motion_planning/blob/master/global_planner/sample_search/rrt.py)            | ![rrt_python.png](assets/rrt_python.png)
**RRT***                 | [![Status](https://img.shields.io/badge/done-v1.0-brightgreen)](https://github.com/ai-winter/python_motion_planning/blob/master/global_planner/sample_search/rrt_star.py)       | ![rrt_star_python.png](assets/rrt_star_python.png)
**Informed RRT**                 | [![Status](https://img.shields.io/badge/done-v1.0-brightgreen)](https://github.com/ai-winter/python_motion_planning/blob/master/global_planner/sample_search/informed_rrt.py)   | ![informed_rrt_python.png](assets/informed_rrt_python.png)
**RRT-Connect**                | [![Status](https://img.shields.io/badge/done-v1.0-brightgreen)](https://github.com/ai-winter/python_motion_planning/blob/master/global_planner/sample_search/rrt_connect.py)    | ![rrt_connect_python.png](assets/rrt_connect_python.png)
| **ACO** | [![Status](https://img.shields.io/badge/done-v1.0-brightgreen)](https://github.com/ai-winter/python_motion_planning/blob/master/global_planner/evolutionary_search/aco.py)      | ![aco_python.png](assets/aco_python.png)
| **GA**  | ![Status](https://img.shields.io/badge/develop-v1.0-red)                                                                                                                        | ![Status](https://img.shields.io/badge/gif-none-yellow) 
| **PSO**  | [![Status](https://img.shields.io/badge/done-v1.0-brightgreen)](https://github.com/ai-winter/python_motion_planning/blob/master/global_planner/evolutionary_search/pso.py)      | ![pso_python.png](assets/pso_python.svg) ![pso_python_cost.png](assets/pso_python_cost.svg) 


## Local Planner

| Planner     | Version                                                                                                                                                | Animation                                     
|-------------|--------------------------------------------------------------------------------------------------------------------------------------------------------| -------------------------------------------------- 
| **PID**     | [![Status](https://img.shields.io/badge/done-v1.0-brightgreen)](https://github.com/ai-winter/python_motion_planning/blob/master/local_planner/pid.py)  | ![pid_python.svg](assets/pid_python.svg) 
| **APF**     | [![Status](https://img.shields.io/badge/done-v1.0-brightgreen)](https://github.com/ai-winter/python_motion_planning/blob/master/local_planner/apf.py)  | ![apf_python.svg](assets/apf_python.svg) 
| **DWA**     | [![Status](https://img.shields.io/badge/done-v1.0-brightgreen)](https://github.com/ai-winter/python_motion_planning/blob/master/local_planner/dwa.py)  | ![dwa_python.svg](assets/dwa_python.svg)
| **RPP**     | [![Status](https://img.shields.io/badge/done-v1.0-brightgreen)](https://github.com/ai-winter/python_motion_planning/blob/master/local_planner/rpp.py)  | ![rpp_python.svg](assets/rpp_python.svg)
| **LQR**     | [![Status](https://img.shields.io/badge/done-v1.0-brightgreen)](https://github.com/ai-winter/python_motion_planning/blob/master/local_planner/lqr.py)  | ![lqr_python.svg](assets/lqr_python.svg) 
| **TEB**     | ![Status](https://img.shields.io/badge/develop-v1.0-red)                                                                                               | ![Status](https://img.shields.io/badge/gif-none-yellow) 
| **MPC**     | [![Status](https://img.shields.io/badge/done-v1.0-brightgreen)](https://github.com/ai-winter/python_motion_planning/blob/master/local_planner/mpc.py)  | ![mpc_python.svg](assets/mpc_python.svg)
| **MPPI**    | ![Status](https://img.shields.io/badge/develop-v1.0-red)                                                                                               |![Status](https://img.shields.io/badge/gif-none-yellow)
| **Lattice** | ![Status](https://img.shields.io/badge/develop-v1.0-red)                                                                                               |![Status](https://img.shields.io/badge/gif-none-yellow)
| **DQN**    | ![Status](https://img.shields.io/badge/develop-v1.0-red)                                                                                               |![Status](https://img.shields.io/badge/gif-none-yellow)
| **DDPG**    | ![Status](https://img.shields.io/badge/develop-v1.0-red)                                                                                               |![Status](https://img.shields.io/badge/gif-none-yellow)

## Curve Generation

| Planner | Version   | Animation                                |
| ------- | -------------------------------------------------------- | -------------------------------------------------------- 
| **Polynomia** | [![Status](https://img.shields.io/badge/done-v1.0-brightgreen)](https://github.com/ai-winter/python_motion_planning/blob/master/curve_generation/polynomial_curve.py) | ![polynomial_curve_python.gif](assets/polynomial_curve_python.gif)
| **Bezier** | [![Status](https://img.shields.io/badge/done-v1.0-brightgreen)](https://github.com/ai-winter/python_motion_planning/blob/master/curve_generation/bezier_curve.py) | ![bezier_curve_python.png](assets/bezier_curve_python.png)
| **Cubic Spline** | [![Status](https://img.shields.io/badge/done-v1.0-brightgreen)](https://github.com/ai-winter/python_motion_planning/blob/master/curve_generation/cubic_spline.py) | ![cubic_spline_python.png](assets/cubic_spline_python.png)
| **BSpline** | [![Status](https://img.shields.io/badge/done-v1.0-brightgreen)](https://github.com/ai-winter/python_motion_planning/blob/master/curve_generation/bspline_curve.py) | ![bspline_curve_python.png](assets/bspline_curve_python.png)
| **Dubins** | [![Status](https://img.shields.io/badge/done-v1.0-brightgreen)](https://github.com/ai-winter/python_motion_planning/blob/master/curve_generation/dubins_curve.py) | ![dubins_curve_python.png](assets/dubins_curve_python.png)
| **Reeds-Shepp** | [![Status](https://img.shields.io/badge/done-v1.0-brightgreen)](https://github.com/ai-winter/python_motion_planning/blob/master/curve_generation/reeds_shepp.py) | ![reeds_shepp_python.png](assets/reeds_shepp_python.gif)
| **Fem-Pos Smoother** | [![Status](https://img.shields.io/badge/done-v1.0-brightgreen)](https://github.com/ai-winter/python_motion_planning/blob/master/curve_generation/fem_pos_smooth.py) | ![fem_pos_smoother_python.png](assets/fem_pos_smoother_python.png)




# Papers
## Global Planning

* [A*: ](https://ieeexplore.ieee.org/document/4082128) A Formal Basis for the heuristic Determination of Minimum Cost Paths
* [JPS:](https://ojs.aaai.org/index.php/AAAI/article/view/7994) Online Graph Pruning for Pathfinding On Grid Maps
* [Lifelong Planning A*: ](https://www.cs.cmu.edu/~maxim/files/aij04.pdf) Lifelong Planning A*
* [D*: ](http://web.mit.edu/16.412j/www/html/papers/original_dstar_icra94.pdf) Optimal and Efficient Path Planning for Partially-Known Environments
* [D* Lite: ](http://idm-lab.org/bib/abstracts/papers/aaai02b.pdf) D* Lite
* [Theta*: ](https://www.jair.org/index.php/jair/article/view/10676) Theta*: Any-Angle Path Planning on Grids
* [Lazy Theta*: ](https://ojs.aaai.org/index.php/AAAI/article/view/7566) Lazy Theta*: Any-Angle Path Planning and Path Length Analysis in 3D
* [S-Theta*: ](https://link.springer.com/chapter/10.1007/978-1-4471-4739-8_8) S-Theta*: low steering path-planning algorithm
* [Anya: ](http://www.grastien.net/ban/articles/hgoa-jair16.pdf) Optimal Any-Angle Pathfinding In Practice
* [RRT: ](http://msl.cs.uiuc.edu/~lavalle/papers/Lav98c.pdf) Rapidly-Exploring Random Trees: A New Tool for Path Planning
* [RRT-Connect: ](http://www-cgi.cs.cmu.edu/afs/cs/academic/class/15494-s12/readings/kuffner_icra2000.pdf) RRT-Connect: An Efficient Approach to Single-Query Path Planning
* [RRT*: ](https://journals.sagepub.com/doi/abs/10.1177/0278364911406761) Sampling-based algorithms for optimal motion planning
* [Informed RRT*: ](https://arxiv.org/abs/1404.2334) Optimal Sampling-based Path Planning Focused via Direct Sampling of an Admissible Ellipsoidal heuristic
* [ACO: ](http://www.cs.yale.edu/homes/lans/readings/routing/dorigo-ants-1999.pdf) Ant Colony Optimization: A New Meta-Heuristic

## Local Planning

* [DWA: ](https://www.ri.cmu.edu/pub_files/pub1/fox_dieter_1997_1/fox_dieter_1997_1.pdf) The Dynamic Window Approach to Collision Avoidance
* [APF: ](https://ieeexplore.ieee.org/document/1087247) Real-time obstacle avoidance for manipulators and mobile robots
* [RPP: ](https://arxiv.org/pdf/2305.20026.pdf) Regulated Pure Pursuit for Robot Path Tracking
* [DDPG: ](https://arxiv.org/abs/1509.02971) Continuous control with deep reinforcement learning

## Curve Generation

* [Dubins: ]() On curves of minimal length with a constraint on average curvature, and with prescribed initial and terminal positions and tangents

# Contact

Maintainers:

* [@omigeft](https://github.com/omigeft) (Wu Maojia)
* [@ai-winter](https://github.com/ai-winter) (Yang Haodong)

You can contact us via the information provided on our profile.

# Acknowledgment

* Our visualization and animation framework of Python Version refers to [https://github.com/zhm-real/PathPlanning](https://github.com/zhm-real/PathPlanning). Thanks sincerely.
