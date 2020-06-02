import numpy as np
from obstacle_environment import ObstacleEnvironment2D
from core.dynamics.linear_system_dynamics import LinearSystemDynamics
import gurobipy as gp
from gurobipy import GRB

TESTA = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [-1, 0, 0, 0],
    [0, 1, 0, 0],
])

TESTB = np.array([
    [0, 0],
    [0, 0],
    [1, 0],
    [0, 1]
])


class MIPPlanner:

    def __init__(self, env: ObstacleEnvironment2D, dyn,
                 T=200, h_k=1e-3):
        self.env = env
        self.dyn = dyn
        self.T = T
        self.h_k = h_k
        self.last_x = None
        self.last_u = None
        self.last_m = None
        self.AbCvx = [(a, b) for a, b in zip(*self.env.get_cvx_ineqs())]
        self.num_regions = len(self.AbCvx)
        self.time_limit = float('inf')

    def get_discrete_variables(self):
        '''
        Each discerete variable corresponds to one convex region
        Each Discrete variable has a key (int) and a value (np.array)
        :return: List of dictionary of discrete variables
        '''
        return {1: np.zeros([2]), 2: np.ones([2])}

    def set_optim_timelimit(self, seconds):
        self.time_limit = seconds

    def optimize_path(self, active_set=None, inactive_set_val=None,
                      initial_soln=None, verbose=False):
        '''

        :param initial_soln: Initial guess for solution to optimization problem
        :param active_set: Every variable in the active set will be optimized over
        Every Variable not in the active set will be set to the constant as
        provided in inital_soln
        :return: Waypoints corresponding to path
        '''
        m = gp.Model('Planner')
        m.Params.timeLimit = self.time_limit
        x, u, q, cvx_constraints, dyn_constraints \
            = (list(), list(), list(), list(), list())
        if active_set is not None:
            assert active_set.dtype == bool, "active set must be np array of bools"
        else:
            active_set = np.ones((self.T, len(self.env.get_cvx_ineqs()[0])),
                                 dtype=bool)
        if inactive_set_val is not None:
            assert inactive_set_val.dtype == bool, "inactive_set_value must be bools"

        for t in range(self.T):
            x.append(m.addMVar(4, lb=-GRB.INFINITY,
                               vtype=GRB.CONTINUOUS,
                               name=f"x_{t}"))
            u.append(m.addMVar(2, lb=-GRB.INFINITY,
                               vtype=GRB.CONTINUOUS,
                               name=f"u_{t}"))
            q.append(x[-1][:2])
            # THis wont work in general it has to be specified as part of
            #one cnvex region in the environment
            # q[-1].setAttr("lb", 0.0)
            # q[-1].setAttr("ub", 100.0)
            # m.addConstr(q[-1] <= 100*np.ones((2,)))
            xbinvars, x_row_constraints = (list(), list())
            const_constraints = list()
            assert np.any(active_set[t]) or np.any(inactive_set_val[t]),\
            "[ERROR] Inconsistent problem definition. Each waypoint must have " \
            "at least one convex region constraint active."

            for region_idx_ in range(self.num_regions):
                if active_set[t, region_idx_]:
                    bin_var, row_constraints = self._add_cvx_region_constraint(
                        q[-1], region_idx_, t, m)
                    xbinvars.append(bin_var)
                    x_row_constraints.append(row_constraints)
                else:
                    if inactive_set_val[t, region_idx_]:

                        const_constraints.append(
                            self._add_cvx_region_constant_constraint(
                                q[-1], region_idx_, t, m))
            if len(const_constraints) == 0 and len(xbinvars) > 0:
                anycvx = m.addConstr(gp.quicksum(xbinvars) >= 1,
                                     name=f"x_{t}_cvx_any")
                cvx_constraints.append((xbinvars, x_row_constraints, None, anycvx))
            else:
                cvx_constraints.append((xbinvars, x_row_constraints, const_constraints, None))
        init_constraint = m.addMConstrs(
            A=np.eye(4),
            x=x[0],
            sense=GRB.EQUAL,
            b=np.array(self.env.start))
        end_constraint = m.addMConstrs(
            A=np.eye(4),
            x=x[-1],
            sense=GRB.EQUAL,
            b=np.array(self.env.end))

        for t in range(self.T - 1):
            dyn_constraints.append(
                m.addConstr(
                    x[t + 1] - x[t] == \
                    0.5 * self.h_k * ( \
                            (self.dyn.A @ x[t] + self.dyn.A @ x[t + 1]) + \
                            (self.dyn.B @ u[t] + self.dyn.B @ u[t + 1]))))

        m.setObjective(sum([ut @ np.eye(2) @ ut for ut in u]), GRB.MINIMIZE)

        m.optimize()
        if m.getAttr('Status') == GRB.INFEASIBLE:
            print('[WARNING] Model provided is infeasible. CVX regions '
                  'probably wrong.')
        elif m.getAttr('Status') == GRB.INF_OR_UNBD:
            assert False, "[ERROR] Should never happen."
        elif m.getAttr('Status') == GRB.TIME_LIMIT:
            print('[INFO] MIP time-limit termination.')

        self.last_x = x
        self.last_u = u
        self.last_m = m
        runtime = m.getAttr('Runtime')
        xnp = [xt.getAttr('X') for xt in x]
        unp = [ut.getAttr('X') for ut in u]
        return xnp, unp, runtime

    def _add_cvx_region_constant_constraint(self, qt, region_idx, t, model):
        Ai = self.AbCvx[region_idx][0]
        bi = self.AbCvx[region_idx][1].squeeze()
        return model.addConstr(Ai@qt <= bi, name=f"x_{t}_cvx_{region_idx}_FON")

    def _add_cvx_region_constraint(self, qt, region_idx, t, model):
        Ai = self.AbCvx[region_idx][0]
        bi = self.AbCvx[region_idx][1]
        binvar = model.addVar(vtype=GRB.BINARY, name=f"x_{t}_cvx_{region_idx}")
        row_constraints = list()
        for row_i in range(Ai.shape[1]):
            row_constraints.append(
                model.addGenConstrIndicator(
                    binvar=binvar,
                    binval=1,
                    lhs=gp.LinExpr(Ai[row_i], qt.vararr),
                    sense=GRB.LESS_EQUAL,
                    rhs=bi[row_i],
                    name=f"x_{t}_cvx_{region_idx}_c_r{row_i}"))
        return binvar, row_constraints
