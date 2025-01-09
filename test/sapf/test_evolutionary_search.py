"""
@file: test_evolutionary_search.py
@breif: path planning based on evolutionary searching application examples
@author: Winter
@update: 2024.4.22
"""
import os

from python_motion_planning.common.structure import Grid
from python_motion_planning.planner import PlannerFactory
from python_motion_planning.utils.logger import ParamsManager
from python_motion_planning.common.utils import Visualizer

if __name__ == '__main__':
    '''
    build environment
    '''
    config_file = os.path.abspath(os.path.join(
        __file__, "../../../config/user_params/user_config.yaml"
    ))
    params = ParamsManager(config_file).getParams()
    env = Grid(params)

    '''
    evolutionary search
    '''
    planner_factory = PlannerFactory()
    planner = planner_factory(params["strategy"]["planner"]["name"], params=params)

    # plan
    cost, path, cost_curve = planner.plan()

    '''
    Visualization
    '''
    visualizer = Visualizer("test_path_planning")
    visualizer.plotEnv(env)
    visualizer.setTitle(str(planner) + "\ncost: " + str(cost))
    # visualizer.plotPoint(start[0], start[1], "s", "#ff0000")
    # visualizer.plotPoint(goal[0], goal[1], "s", "#1155cc")
    visualizer.plotPath(path)
    # visualizer.plotPoint(start[0], start[1], "s", "#ff0000")
    # visualizer.plotPoint(goal[0], goal[1], "s", "#1155cc")
    visualizer.plotCurve(
        xlist=[[i + 1 for i in range(len(cost_curve))]],
        ylist=[cost_curve],
        xlabels=["iterations"],
        ylabels=["cost"],
        fig_name="cost_curve"
    )

    visualizer.show()
