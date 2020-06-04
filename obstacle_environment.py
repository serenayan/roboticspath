import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from matplotlib import collections as mc

# Constants I will use testing
RED = np.array([1, 0, 0, 1])
BLUE = np.array([0, 0, 1, 1])
GREEN = np.array([0, 1, 0, 1])
BLACK = np.array([0, 0, 0, 1])

START = (5, 5, 50, 0)
END = (90, 90, 0, 0)
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
    def __init__(self, start, end, obstacles, a, b):
        self.start = start
        self.end = end
        self.obstacles = obstacles
        self.a = a
        self.b = b

    def get_cvx_ineqs(self):
        '''
        Converts convex sets from whatever format to Ax <= b
        :return: A and b Matrices
        '''
        return (self.a, self.b)

    def plot_path(self, waypoints, ax=None):
        '''

        :param ax:
        :param waypoints: List of 2D np.array()
        :return:
        '''
        # TODO: Given set of waypoints plot a path
        if ax is None:
            ax = self.plot(False)

        for value in waypoints:
            if len(value) == 2:
                ax.plot(value[0], value[1], 'bo')
            else:
                vnorm = norm((value[2], value[3]))
                vnorm = vnorm if vnorm > 0 else 1
                ax.arrow(value[0], value[1], value[2]/vnorm, value[3]/vnorm,
                         width=0.5,
                         length_includes_head=True)
        return ax

    def plot(self, show=True):
        lines = OUTERBOUNDS + self.obstacles + self.cvx_regions
        colors = [BLACK] * len(OUTERBOUNDS) + \
                 [RED] * len(self.obstacles) + \
                 [BLUE] * len(self.cvx_regions)
        lc = mc.LineCollection(lines, colors=colors, linewidths=2)
        start_circ = plt.Circle(self.end[:2], 2, color='g')
        end_circ = plt.Circle(self.start[:2], 2, color='r')
        fig, ax = plt.subplots()
        ax.set_xlim(-20, 120)
        ax.set_ylim(-20, 120)
        ax.add_collection(lc)
        ax.add_artist(start_circ)
        ax.add_artist(end_circ)
        ######

        ######
        if show:
            plt.show()
        return ax