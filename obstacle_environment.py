import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections as mc

# Constants I will use testing
RED = np.array([1, 0, 0, 1])
BLUE = np.array([0, 0, 1, 1])
GREEN = np.array([0, 1, 0, 1])
BLACK = np.array([0, 0, 0, 1])

START = (5, 5)
END = (90, 90)
# these are assumed so don't pass them in CVX_INEQS
OUTERBOUNDS = [
    [(0, 0), (0, 100)],
    [(0, 0), (100, 0)],
    [(100, 0), (100, 100)],
    [(0, 100), (100, 100)]
]
OBSTACLES = [
    [(10, 10), (60, 10)],
    [(10, 10), (10, 90)],
    [(60, 10), (10, 90)],
]
CVX_LINES = [
    [(10, 0), (10, 100)],
    [(0, 10), (100, 10)],
    [(0, 90), (100, 90)],
    [(60, 0), (60, 100)],
]
A_MATS = [
    np.eye(2),
    np.array([[0, 0], [0, 1]]),
    np.array([[0, 0], [0, -1]]),
    np.array([[-1, 0], [0, 0]])
]
B_MATS = [
    np.array([[10], [90]]),
    np.array([[0], [10]]),
    np.array([[0], [-90]]),
    np.array([[-60], [0]]),
]


class ObstacleEnvironment2D:
    def __init__(self, start, end, obstacles, cvx_regions):
        self.start = start
        self.end = end
        self.obstacles = obstacles
        self.cvx_regions = cvx_regions

    def get_cvx_ineqs(self):
        '''
        Converts convex sets from whatever format to Ax <= b
        :return: A and b Matrices
        '''
        return (A_MATS, B_MATS)

    def plot_path(self, waypoints):
        '''

        :param waypoints: List of 2D np.array()
        :return:
        '''
        # TODO: Given set of waypoints plot a path
        pass

    def plot(self):
        lines = OUTERBOUNDS + self.obstacles + self.cvx_regions
        colors = [BLACK] * len(OUTERBOUNDS) + \
                 [RED] * len(self.obstacles) + \
                 [BLUE] * len(self.cvx_regions)
        lc = mc.LineCollection(lines, colors=colors, linewidths=2)
        start_circ = plt.Circle(self.end, 2, color='g')
        end_circ = plt.Circle(self.start, 2, color='r')
        fig, ax = plt.subplots()
        ax.set_xlim(-1, 101)
        ax.set_ylim(-1, 101)
        ax.add_collection(lc)
        ax.add_artist(start_circ)
        ax.add_artist(end_circ)
        plt.show()
        pass


START_ENV = ObstacleEnvironment2D(START, END, OBSTACLES, CVX_LINES)
