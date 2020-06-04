import numpy as np
from mip_planner import MIPPlanner, TESTA, TESTB
from obstacle_environment import DEFAULT_ENV
from core.dynamics.linear_system_dynamics import LinearSystemDynamics
from collections import namedtuple

OptimState = namedtuple("OptimState", "x, u, aset, iset_val")
Action = namedtuple("Action", "aset, iset_val")


class GymOptEnv:
    def __init__(self):
        self.planner = MIPPlanner(DEFAULT_ENV, LinearSystemDynamics(TESTA, TESTB))
        self.T = self.planner.T
        self.NCVX = len(self.planner.env.get_cvx_ineqs())
        self.random_seed = 0
        self.state = OptimState(
            np.zeros((self.T, self.planner.dyn.n)),
            np.zeros((self.T, self.planner.dyn.m)),
            np.ones((self.T, self.planner.env.get_num_cvx()), dtype=bool),
            np.ones((self.T, self.planner.env.get_num_cvx()), dtype=bool))
        # generate ground truth
        self.x_true, \
        self.u_true, \
        self.obj_true, \
        self.aset_true, \
        self.iset_val_true, \
        self.rt_true = self.planner.optimize_path(
            self.state.aset,
            self.state.iset_val,
            (self.state.x, self.state.u))
        # After getting true value limit runtime
        self.planner.set_optim_timelimit(5)

    def step(self, action: Action):
        '''

        :param action:
        :return:
        '''
        x_new, u_new, obj_new, aset, iset_val, rt = \
            self.planner.optimize_path(action.aset, action.iset_val,
                                       (self.state.x, self.state.u))

        self.state = OptimState(x_new, u_new, aset, iset_val)
        reward = obj_new - self.obj_true
        assert reward <= 0  # min obj <= obj
        # TODO: make tolerance reasonable
        done = np.isclose(obj_new, self.obj_true, rtol=0.01)
        return self.state, reward, done

    def seed(self, s):
        self.random_seed = s

    def reset(self):
        self.state = OptimState(
            np.zeros((self.T, self.planner.dyn.n)),
            np.zeros((self.T, self.planner.dyn.m)),
            np.ones((self.T, self.planner.env.get_num_cvx()), dtype=bool),
            np.ones((self.T, self.planner.env.get_num_cvx()), dtype=bool))
        return self.state
