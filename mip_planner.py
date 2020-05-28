import numpy as np
from obstacle_environment import ObstacleEnvironment2D
from core.dynamics.linear_system_dynamics import LinearSystemDynamics
import gurobipy

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

    def __init__(self, env: ObstacleEnvironment2D, dynamics):
        self.env = env
        self.dynamics = dynamics

    def get_discrete_variables(self):
        '''
        Each discerete variable corresponds to one convex region
        Each Discrete variable has a key (int) and a value (np.array)
        :return: List of dictionary of discrete variables
        '''
        return {1: np.zeros([2]), 2: np.ones([2])}

    def optimize_path(self, initial_soln, active_set):
        '''

        :param initial_soln: Initial guess for solution to optimization problem
        :param active_set: Every variable in the active set will be optimized over
        Every Variable not in the active set will be set to the constant as
        provided in inital_soln
        :return: Waypoints corresponding to path
        '''
        pass