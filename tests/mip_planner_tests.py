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
    xs, us = dyn.simulate(START_ENV.start + (100, 0), ctrl,
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
    x, u = planner.optimize_path()

    for v in planner.last_m.getVars():
        print(f'{v.varName}={v.x}')

    START_ENV.plot_path(x)
    plt.show()

