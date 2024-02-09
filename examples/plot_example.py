from python_motion_planning import AStar, RRT, ACO, PID, Grid, Map, Dubins

# plt = AStar(start=(5, 5), goal=(45, 25), env=Grid(51, 31))
# plt = RRT(start=(18, 8), goal=(37, 18), env=Map(51, 31), max_dist=0.5, sample_num=10000)
# plt = ACO(start=(5, 5), goal=(45, 25), env=Grid(51, 31))
# plt = PID(start=(5, 5, 0), goal=(45, 25, 0), env=Grid(51, 31))
#
# plt.run()

points = [(0, 0, 0), (10, 10, -90), (20, 5, 60), (30, 10, 120),
          (35, -5, 30), (25, -10, -120), (15, -15, 100), (0, -10, -90)]
generator = Dubins(step=0.1, max_curv=0.25)
generator.run(points)