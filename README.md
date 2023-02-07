
# Introduction

`Motion planning` plans the state sequence of the robot without conflict between the start and goal. 

`Motion planning` mainly includes `Path planning` and `Trajectory planning`.

* `Path Planning`: It's based on path constraints (such as obstacles), planning the optimal path sequence for the robot to travel without conflict between the start and goal.
* `Trajectory planning`: It plans the motion state to approach the global path based on kinematics, dynamics constraints and path sequence.

This repository provides the implement of common `Motion planning` algorithm, welcome your star & fork & PR.

The theory analysis can be found at [motion-planning](https://blog.csdn.net/frigidwinter/category_11410243.html)

We also provide ROS C++ version at [https://github.com/ai-winter/ros_motion_planning](https://github.com/ai-winter/ros_motion_planning) and Matlab Version at [https://github.com/ai-winter/matlab_motion_planning](https://github.com/ai-winter/matlab_motion_planning)

# Quick Start
The file structure is shown below

```
├─gif
├─graph_search
├─sample_search
├─utils
└─main.py
```
The global planning algorithm implementation is in the folder `graph_search` and `sample_search`; The local planning algorithm implementation is in the folder `local_planner`.

To start simulation, open `./main.py` and select the algorithm, for example

```python
if __name__ == '__main__':
    '''
    sample search
    '''
    # build environment
    start = (18, 8)
    goal = (37, 18)
    env = Map(51, 31)

    planner = InformedRRT(start, goal, env, max_dist=0.5, r=12, sample_num=1500)

    # animation
    planner.run()
```

# Version
## Global Planner

Planner      |   Version   | Animation
------------ | --------- | --------- 
**GBFS**              | [![Status](https://img.shields.io/badge/done-v1.0-brightgreen)](https://github.com/ai-winter/python_motion_plannning/blob/master/graph_search/gbfs.py)   | ![gbfs_python.png](gif/gbfs_python.png) 
**Dijkstra**                 | [![Status](https://img.shields.io/badge/done-v1.0-brightgreen)](https://github.com/ai-winter/python_motion_plannning/blob/master/graph_search/dijkstra.py) | ![dijkstra_python.png](gif/dijkstra_python.png)
**A***               | ![Status](https://img.shields.io/badge/done-v1.0-brightgreen) |  ![a_star_python.png](gif/a_star_python.png) 
**JPS**                 | [![Status](https://img.shields.io/badge/done-v1.0-brightgreen)](https://github.com/ai-winter/python_motion_plannning/blob/master/graph_search/jps.py) | ![jps_python.png](gif/jps_python.png)
**D***                  | [![Status](https://img.shields.io/badge/done-v1.0-brightgreen)](https://github.com/ai-winter/python_motion_plannning/blob/master/graph_search/d_star.py) | ![d_star_python.png](gif/d_star_python.png)
**LPA***                 | [![Status](https://img.shields.io/badge/done-v1.0-brightgreen)](https://github.com/ai-winter/python_motion_plannning/blob/master/graph_search/lpa_star.py) | ![lpa_star_python.png](gif/lpa_star_python.png) 
**D\* Lite**                | [![Status](https://img.shields.io/badge/done-v1.0-brightgreen)]((https://github.com/ai-winter/python_motion_plannning/blob/master/graph_search/d_star_lite.py)) | ![d_star_lite_python.png](gif/d_star_lite_python.png) 
**RRT**                 | [![Status](https://img.shields.io/badge/done-v1.0-brightgreen)](https://github.com/ai-winter/python_motion_plannning/blob/master/sample_search/rrt.py) | ![rrt_python.png](gif/rrt_python.png)
**RRT***                 | [![Status](https://img.shields.io/badge/done-v1.0-brightgreen)](https://github.com/ai-winter/python_motion_plannning/blob/master/sample_search/rrt_star.py) | ![rrt_star_python.png](gif/rrt_star_python.png)
**Informed RRT**                 | [![Status](https://img.shields.io/badge/done-v1.0-brightgreen)](https://github.com/ai-winter/python_motion_plannning/blob/master/sample_search/informed_rrt.py) | ![informed_rrt_python.png](gif/informed_rrt_python.png)
**RRT-Connect**                | [![Status](https://img.shields.io/badge/done-v1.0-brightgreen)](https://github.com/ai-winter/python_motion_plannning/blob/master/sample_search/rrt_connect.py) | ![rrt_connect_python.png](gif/rrt_connect_python.png)

## Local Planner
| Planner |  Version   | Animation                                     
| ------- | ---------------------------------------- | -------------------------------------------------- 
| **PID** | ![Status](https://img.shields.io/badge/develop-v1.0-red) |![Status](https://img.shields.io/badge/gif-none-yellow) 
| **APF** | ![Status](https://img.shields.io/badge/develop-v1.0-red) |![Status](https://img.shields.io/badge/gif-none-yellow) 
| **DWA** |  ![Status](https://img.shields.io/badge/develop-v1.0-red) | ![Status](https://img.shields.io/badge/gif-none-yellow) 
| **TEB** | ![Status](https://img.shields.io/badge/develop-v1.0-red) | ![Status](https://img.shields.io/badge/gif-none-yellow) 
| **MPC** | ![Status](https://img.shields.io/badge/develop-v1.0-red) | ![Status](https://img.shields.io/badge/gif-none-yellow) 
| **Lattice** | ![Status](https://img.shields.io/badge/develop-v1.0-red) |![Status](https://img.shields.io/badge/gif-none-yellow) 

## Intelligent Algorithm

| Planner | Version   | Animation                                |
| ------- | -------------------------------------------------------- | -------------------------------------------------------- 
| **ACO** | ![Status](https://img.shields.io/badge/develop-v1.0-red) | ![Status](https://img.shields.io/badge/gif-none-yellow) 
| **GA**  | ![Status](https://img.shields.io/badge/develop-v1.0-red) | ![Status](https://img.shields.io/badge/gif-none-yellow) 
| **PSO**  | ![Status](https://img.shields.io/badge/develop-v1.0-red) | ![Status](https://img.shields.io/badge/gif-none-yellow) 
| **ABC** | ![Status](https://img.shields.io/badge/develop-v1.0-red) | ![Status](https://img.shields.io/badge/gif-none-yellow) 





# Papers
## Search-based Planning
* [A*: ](https://ieeexplore.ieee.org/document/4082128) A Formal Basis for the heuristic Determination of Minimum Cost Paths
* [JPS:](https://ojs.aaai.org/index.php/AAAI/article/view/7994) Online Graph Pruning for Pathfinding On Grid Maps
* [Lifelong Planning A*: ](https://www.cs.cmu.edu/~maxim/files/aij04.pdf) Lifelong Planning A*
* [D*: ](http://web.mit.edu/16.412j/www/html/papers/original_dstar_icra94.pdf) Optimal and Efficient Path Planning for Partially-Known Environments
* [D* Lite: ](http://idm-lab.org/bib/abstracts/papers/aaai02b.pdf) D* Lite

## Sample-based Planning
* [RRT: ](http://msl.cs.uiuc.edu/~lavalle/papers/Lav98c.pdf) Rapidly-Exploring Random Trees: A New Tool for Path Planning
* [RRT-Connect: ](http://www-cgi.cs.cmu.edu/afs/cs/academic/class/15494-s12/readings/kuffner_icra2000.pdf) RRT-Connect: An Efficient Approach to Single-Query Path Planning
* [RRT*: ](https://journals.sagepub.com/doi/abs/10.1177/0278364911406761) Sampling-based algorithms for optimal motion planning
* [Informed RRT*: ](https://arxiv.org/abs/1404.2334) Optimal Sampling-based Path Planning Focused via Direct Sampling of an Admissible Ellipsoidal heuristic

## Local Planning

* [DWA: ](https://www.ri.cmu.edu/pub_files/pub1/fox_dieter_1997_1/fox_dieter_1997_1.pdf) The Dynamic Window Approach to Collision Avoidance




# Acknowledgment
* Our visualization and animation framework of Python Version refers to [https://github.com/zhm-real/PathPlanning](https://github.com/zhm-real/PathPlanning). Thanks sincerely.