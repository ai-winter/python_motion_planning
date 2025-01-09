"""
@file: test_controller.py
@breif: local controller application examples
@author: Winter
@update: 2024.4.23
"""
import os

from python_motion_planning.common.structure import Grid
from python_motion_planning.controller import ControlFactory
from python_motion_planning.planner import PlannerFactory
from python_motion_planning.common.utils import Visualizer
from python_motion_planning.common.utils import ParamsManager
from python_motion_planning.common.geometry import Point3d

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
    planner_factory = PlannerFactory()
    planner = planner_factory(params["strategy"]["planner"]["name"], params=params)
    start = params["agents"][0]["start"]
    start = Point3d(start[0], start[1], start[2])
    goal = params["agents"][0]["goal"]
    goal = Point3d(goal[0], goal[1], goal[2])
    path, output = planner.plan(start, goal)

    '''
    path tracking
    '''
    control_factory = ControlFactory()
    controller = control_factory(params["strategy"]["controller"]["name"], params=params)
    output = controller.plan(path)

    '''
    Visualization
    '''
    visualizer = Visualizer("test_path_tracking")
    visualizer.plotGridMap(env)
    visualizer.plotPath(path, name="normal", props={"style": "--"})
    visualizer.plotGrids([
        {"x": start.x(), "y": start.y(), "name": "start"},
        {"x": goal.x(), "y": goal.y(), "name": "goal"},
    ])

    cost, frame_info = 0, []
    for info in output:
        if info["type"] == "value" and info["name"] == "success":
            if info["data"] == False:
                quit
        if info["type"] == "value" and info["name"] == "cost":
            cost = info["data"]
        if info["type"] == "frame":
            props = info["props"] if "props" in info.keys() else {}
            frame_info.append({"name": info["name"], "data": info["data"], "props": props})

    visualizer.setTitle("Path planning: " + str(planner) + "\nController: " \
        + str(controller) + "\ncost: " + str(cost))
    visualizer.plotFrames(frame_info)

    visualizer.show()