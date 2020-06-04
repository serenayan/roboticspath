import pytest

import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from mip_planner import MIPPlanner, TESTA, TESTB
from obstacle_environment import START_ENV
from core.dynamics.linear_system_dynamics import LinearSystemDynamics
from core.controllers import ConstantController

def test_dynamics_integration():
    dyn = LinearSystemDynamics(TESTA, TESTB)
    ctrl = ConstantController(dyn, np.array([0, 0]))
    xs, us = dyn.simulate(START_ENV.start, ctrl,
                          np.linspace(0, 10, 100))
    #enable to see plot of dynamics
    #START_ENV.plot_path(xs)
    # plt.show()

def grouped(iterable, n):
    return zip(*[iter(iterable)]*n)

def test_mip_planner():
    planner = MIPPlanner(START_ENV, LinearSystemDynamics(TESTA, TESTB),
                         T=200,
                         h_k=0.001)
    x, u, _, _, _, _ = planner.optimize_path()

    np.testing.assert_allclose(x[0], START_ENV.start)
    np.testing.assert_allclose(x[-1], START_ENV.end)
    assert x[70][1] < 5 # bottom right corner
    # for v in planner.last_m.getVars():
    #     print(f'{v.varName}={v.x}')
    # START_ENV.plot_path(x)
    # plt.show()

def test_mip_discrete_regions():
    planner = MIPPlanner(START_ENV, LinearSystemDynamics(TESTA, TESTB),
                         T=200,
                         h_k=0.001)
    n_regions = len(START_ENV.get_cvx_ineqs()[0])
    n_timesteps = planner.T
    active_set = np.zeros((n_timesteps, n_regions), dtype=bool)
    active_set[:100, 0] = True
    active_set[100:, 2] = True
    active_set[120:, 3] = True
    active_set_vals = active_set

    x, u, _, _, _, _ = planner.optimize_path(active_set, active_set_vals)
    np.testing.assert_allclose(x[0], START_ENV.start)
    np.testing.assert_allclose(x[-1], START_ENV.end)
    assert x[90][0] < 10 and x[90][1] > 80 # top left corner
    # START_ENV.plot_path(x)
    # plt.show()
    # pass

def test_mip_initalization_given_active():
    planner = MIPPlanner(START_ENV, LinearSystemDynamics(TESTA, TESTB),
                         T=200,
                         h_k=0.001)
    n_regions = len(START_ENV.get_cvx_ineqs()[0])
    n_timesteps = planner.T
    active_set = np.zeros((n_timesteps, n_regions), dtype=bool)
    active_set[:100, 0] = True
    active_set[100:, 2] = True
    active_set[120:, 3] = True
    active_set_vals = active_set.copy()
    x, u, obj, active_set, inactive_set_val, rt = planner.optimize_path(
        active_set, active_set_vals)

    x_inited, u_inited, obj_inited, as_inited,\
    is_val_inited, rt_inited = \
        planner.optimize_path(np.zeros_like(active_set, dtype=bool),
                              inactive_set_val,
                              initial_soln=(x, u))
    # ax = START_ENV.plot_path(x)
    # START_ENV.plot_path(x_inited, ax)
    # plt.show()
    np.testing.assert_allclose(np.stack(x), np.stack(x_inited))
    np.testing.assert_allclose(np.stack(u), np.stack(u_inited))
    np.testing.assert_allclose(inactive_set_val, is_val_inited)
    np.testing.assert_allclose([obj], [obj_inited])
    assert rt_inited < rt

def test_mip_initalization_default():
    planner = MIPPlanner(START_ENV, LinearSystemDynamics(TESTA, TESTB),
                         T=200,
                         h_k=0.001)
    n_regions = len(START_ENV.get_cvx_ineqs()[0])
    n_timesteps = planner.T
    # active_set = np.zeros((n_timesteps, n_regions), dtype=bool)
    # active_set[:100, 0] = True
    # active_set[100:, 2] = True
    # active_set[120:, 3] = True
    # active_set_vals = active_set.copy()
    active_set = None
    active_set_vals = None
    x, u, obj, active_set, inactive_set_val, rt = planner.optimize_path(
        active_set, active_set_vals)

    x_inited, u_inited, obj_inited, as_inited,\
    is_val_inited, rt_inited = \
        planner.optimize_path(np.zeros_like(active_set, dtype=bool),
                              inactive_set_val,
                              initial_soln=(x, u))
    # ax = START_ENV.plot_path(x)
    # START_ENV.plot_path(x_inited, ax)
    # plt.show()
    np.testing.assert_allclose(np.stack(x), np.stack(x_inited))
    np.testing.assert_allclose(np.stack(u), np.stack(u_inited))
    np.testing.assert_allclose(inactive_set_val, is_val_inited)
    np.testing.assert_allclose([obj], [obj_inited])
    assert rt_inited < rt