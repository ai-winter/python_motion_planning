"""
@file: test_path_planner.py
@breif: path planning application examples
@author: Winter
@update: 2024.4.22
"""
import os

from python_motion_planning.common.structure import Grid
from python_motion_planning.common.geometry import Point3d
from python_motion_planning.common.utils import ParamsManager
from python_motion_planning.common.utils import Visualizer
from python_motion_planning.path_planner import PathPlannerFactory

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
    path planning
    '''
    planner_factory = PathPlannerFactory()
    planner = planner_factory(params["strategy"]["path_planner"]["name"], env=env, params=params)
    start = params["agents"][0]["start"]
    start = Point3d(start[0], start[1], start[2])
    goal = params["agents"][0]["goal"]
    goal = Point3d(goal[0], goal[1], goal[2])
    path, output = planner.plan(start, goal)

    '''
    Visualization
    '''
    visualizer = Visualizer("test_path_planning")
    visualizer.plotGridMap(env)

    cost = 0
    for info in output:
        if info["type"] == "value" and info["name"] == "success":
            if info["data"] == False:
                quit
        if info["type"] == "value" and info["name"] == "cost":
            cost = info["data"]
        if info["type"] == "path":
            visualizer.plotPath(
                info["data"], name=info["name"], props=info["props"] if "props" in info.keys() else {}
            )
        if info["type"] == "grids" and info["name"] == "expand":
           visualizer.plotGrids([
                {"x": pt.x(), "y": pt.y(), "name": "custom"} for pt in info["data"]
                if ((pt != start) and (pt != goal))
            ])
        if info["type"] == "callback" and info["name"] == "visualization":
            planner.setVisualizer(visualizer)
            visualizer.connect('button_press_event', info["data"])

    visualizer.plotGrids([
        {"x": start.x(), "y": start.y(), "name": "start"},
        {"x": goal.x(), "y": goal.y(), "name": "goal"},
    ])
    visualizer.setTitle(f"{str(planner)}\ncost: {cost}")
    visualizer.show()

