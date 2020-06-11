from env_gen import load_folder_and_plot
from obstacle_environment import ObstacleEnvironment2D
from mip_planner import MIPPlanner, TESTA, TESTB
from core.dynamics.linear_system_dynamics import LinearSystemDynamics
import matplotlib.pyplot as plt
from graph_gen import generate_path_list
import numpy as np


# np.random.seed(1)
# START = (10, 10, 0, 0)
# END = (90, 90, 0, 0)
# OBSTACLES = None
# t = 200
# time_steps = 10
# x_b = (0, 100)
# y_b = (0, 100)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# a, b = load_folder_and_plot(ax, (START[0], START[1]), (END[0], END[1]), "envs/10", x_b, y_b)  # read and plot
# paths = generate_path_list(a, b, START, END)
# print(paths)
# # pick a path
# p = paths[np.random.randint(len(paths))]
# print(p)
# START_ENV = ObstacleEnvironment2D(START, END, OBSTACLES, a, b)
# planner = MIPPlanner(START_ENV, LinearSystemDynamics(TESTA, TESTB),
#                      T=t,
#                      h_k=0.001, presolve=2)
# planner.set_optim_timelimit(600)
# last_inactive_set = np.zeros((t, len(a)),
#                         dtype=bool)
# for row in range(0, int(t/time_steps)):
#     last_inactive_set[row][p[0]] = True
#     last_inactive_set[180 + row][p[-1]] = True
# curr_reg = 0
# wp = None
# ob = None
# runtime = 0
# for step in range(0, time_steps):
#     print("STEP " + str(step))
#     active_set = np.zeros((t, len(a)),
#                          dtype=bool)
#     inactive_set = last_inactive_set.copy()
#     for row in range(0, int(t/time_steps)):
#         active_set[20*step + row] = True
#     for step2 in range(step + 1, time_steps - 1):
#         node_alloc = len(p) - curr_reg - 2
#         if node_alloc > 0:
#             step_alloc = time_steps - step - 2
#             per_node = int(step_alloc / node_alloc)
#             curr_step = step2 - step
#             reg = int(curr_step/per_node)
#             for row in range(0, int(t / time_steps)):
#                 t_reg = min(curr_reg + reg + 1, len(p) - 1)
#                 inactive_set[20 * step2 + row][p[t_reg]] = True
#         else:
#             for row in range(0, int(t / time_steps)):
#                 inactive_set[20 * step2 + row][p[-1]] = True
#     print(active_set)
#     print(inactive_set)
#     wp, _, ob, ac, inac, r = planner.optimize_path(active_set=active_set, inactive_set_val=inactive_set)
#     last_inactive_set = inac.copy()
#     last_reg = np.where(inac[20*step + 19] == True)[0]
#     try:
#         curr_reg = p.index(last_reg)
#     except ValueError:
#         pass
#     runtime += r
# START_ENV.plot_path(wp, ax)
# print(ob)
# print(runtime)
# fig.show()

START = (10, 10, 0, 0)
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
