from env_gen import load_folder_and_plot
from obstacle_environment import ObstacleEnvironment2D
from mip_planner import MIPPlanner, TESTA, TESTB
from core.dynamics.linear_system_dynamics import LinearSystemDynamics
import matplotlib.pyplot as plt

START = (10, 10, 50, 0)
END = (90, 90, 0, 0)
OBSTACLES = None
x_b = (0, 100)
y_b = (0, 100)
fig = plt.figure()
ax = fig.add_subplot(111)
a, b = load_folder_and_plot(ax, (START[0], START[1]), (END[0], END[1]), "envs/10", x_b, y_b)  # read and plot
START_ENV = ObstacleEnvironment2D(START, END, OBSTACLES, a, b)
planner = MIPPlanner(START_ENV, LinearSystemDynamics(TESTA, TESTB),
                     T=200,
                     h_k=0.001)
planner.set_optim_timelimit(900)
wp, _, _, _, _, r = planner.optimize_path()
START_ENV.plot_path(wp, ax)
print(r)
fig.show()