import pytest

import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from mip_planner import TESTA, TESTB
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
    m = gp.Model('Planner')
    dyn = LinearSystemDynamics(TESTA, TESTB)
    T = 200 # number of discrete timesteps
    h_k = 0.001
    x = list()
    u = list()
    q = list()
    cvx_constraints = list()
    # cvxi = list()
    # cvxr = list()
    AbCvx = [(a,b) for a,b in zip(*START_ENV.get_cvx_ineqs())]
    num_regions = len(AbCvx)

    for t in range(T):
        x.append(m.addMVar(4, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"x_{t}"))
        u.append(m.addMVar(2, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS,
                           name=f"u_{t}"))
        q.append(x[-1][:2])
        xbinvars = list()
        xcvx = list()
        #for each xt generate cvx constraints
        for region_idx in range(num_regions):
            Ai = AbCvx[region_idx][0]
            bi = AbCvx[region_idx][1].squeeze()
            binvar = m.addVar(vtype=GRB.BINARY, name=f"x_{t}_cvx_{region_idx}")

            # cvx_r_constraint = [m.addConstr(
            #     (binvar == 1) >> (Ai[row_i] @ q[-1] <= bi[row_i]),
            #     name=f"x_{t}_cvx_{region_idx}_c_r{row_i}") for row_i in
            #     range(Ai.shape[1])]
            cvx_r_constraint = [
                m.addGenConstrIndicator(binvar=binvar,
                                        binval=1,
                                        lhs=gp.LinExpr(Ai[row_i], q[-1].vararr),
                                        sense=GRB.LESS_EQUAL,
                                        rhs=bi[row_i],
                                        name=f"x_{t}_cvx_{region_idx}_c_r{row_i}")
                for row_i in range(Ai.shape[1])]
            xbinvars.append(binvar)
            xcvx.append(cvx_r_constraint)
        anycvx = m.addConstr(gp.quicksum(xbinvars) >= 1, name=f"x_{t}_cvx_any")
        cvx_constraints.append((xbinvars, xcvx, anycvx))

    init_constraint = m.addMConstrs(
        A=np.eye(4),
        x=x[0],
        sense=GRB.EQUAL,
        b=np.array(START_ENV.start + (50, 0)))
    end_constraint = m.addMConstrs(
        A=np.eye(4),
        x=x[-1],
        sense=GRB.EQUAL,
        b=np.array(START_ENV.end + (0, 0)))
    dyn_constraints = list()
    for t in range(T-1):
        xt = x[t]
        ut = u[t]
        utp1 = u[t+1]
        xtp1 = x[t+1]
        dyn_constraints.append(
            m.addConstr(xtp1 - xt == 0.5 * h_k * (dyn.A@xt + dyn.A@xtp1) + (
            dyn.B@ut + dyn.B@utp1)))

    m.setObjective(sum([ut @ np.eye(2) @ ut for ut in u]), GRB.MINIMIZE)
    q = [xt[2:] for xt in x]

    m.optimize()
    for v in m.getVars():
        print(f'{v.varName}={v.x}')

    START_ENV.plot_path([xt.getAttr('X') for xt in x])
    plt.show()
    pass

