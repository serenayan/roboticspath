import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


def gen_line_2_points(x1, y1, x2=None, y2=None, rise=None, run=None, rev=False):
    if rise is None and run is None:
        rise = y2 - y1
        run = x2 - x1
    const = -rise * x1 + run * y1
    if rev:
        return [rise, -run], -const
    else:
        return [-rise, run], const


def gen_triangle_2_points(x1, y1, x2, y2, x_bounds, y_bounds):
    constraints = []
    which_small = bool(random.getrandbits(1))
    width1 = np.random.randint(10, 25) if which_small else np.random.randint(5, 10)
    up_y1 = y1 + width1 + np.random.randint(max([-3, -width1 + 2]), 3)
    down_y1 = y1 - width1 - np.random.randint(max([-3, -width1 + 2]), 3)
    width2 = np.random.randint(5, 10) if which_small else np.random.randint(10, 25)
    up_y2 = y2 + width2 + np.random.randint(max([-3, 2 - width2]), 3)
    down_y2 = y2 - width2 - np.random.randint(max([-3, 2 - width2]), 3)
    a_up, b_up = gen_line_2_points(x1, up_y1, x2=x2, y2=up_y2)
    a_down, b_down = gen_line_2_points(x1, down_y1, x2=x2, y2=down_y2, rev=True)
    constraints.append((a_up, b_up))
    constraints.append((a_down, b_down))
    a_mid, const = gen_line_2_points(x1, y1, x2=x2, y2=y2)
    a = np.array([a_up, a_down])
    b = np.array([b_up, b_down])
    sol = np.linalg.solve(a, b)
    if sol[0] < x1 and sol[0] < x2:
        if x2 > x1:
            new_x = x2 + np.random.randint(3, 7)
        else:
            new_x = x1 + np.random.randint(3, 7)
        rev = True if float(-a_mid[0])/a_mid[1] > 0 else False
    else:
        if x2 < x1:
            new_x = x2 - np.random.randint(3, 7)
        else:
            new_x = x1 - np.random.randint(3, 7)
        rev = False if float(-a_mid[0]) / a_mid[1] > 0 else True
    new_y = (const - a_mid[0] * new_x) / a_mid[1]
    a_3, b_3 = gen_line_2_points(new_x, new_y, rise=a_mid[1], run=a_mid[0], rev=rev)
    constraints.append((a_3, b_3))
    pts = []
    constraints, pts = check_lines_against_bounds(a_up, b_up, a_down, b_down, x_bounds, y_bounds, constraints, pts)
    constraints, pts = check_lines_against_bounds(a_up, b_up, a_3, b_3, x_bounds, y_bounds, constraints, pts)
    constraints, pts = check_lines_against_bounds(a_3, b_3, a_down, b_down, x_bounds, y_bounds, constraints, pts)
    return constraints, pts


def determine_constraint(a1, b1, a2, b2, line_a, line_b, x_bounds, y_bounds, constraints,  points):
    added = False
    int1 = tuple(np.linalg.solve(np.array([a1, line_a]), np.array([b1, line_b])))
    int2 = tuple(np.linalg.solve(np.array([a2, line_a]), np.array([b2, line_b])))
    within1 = point_within_bounds(int1, x_bounds, y_bounds, line_a=line_a)
    within2 = point_within_bounds(int2, x_bounds, y_bounds, line_a=line_a)
    if within1 and int1 not in points:
        points.append(int1)
    if within2 and int2 not in points:
        points.append(int2)
    if within1 or within2:
        eqn = (line_a, line_b)
        if eqn not in constraints:
            constraints.append(eqn)
            added = True
    return constraints, points, added


def check_lines_against_bounds(a1, b1, a2, b2, x_bounds, y_bounds, constrs, pts):
    sol = tuple(np.linalg.solve(np.array([a1, a2]), np.array([b1, b2])))
    if point_within_bounds(sol, x_bounds, y_bounds):
        pts.append(sol)
    ints = []
    if sol[0] < x_bounds[0]:
        constrs, pts, add = determine_constraint(a1, b1, a2, b2, [-1, 0], -x_bounds[0], x_bounds, y_bounds, constrs, pts)
        if add:
            ints.append(x_bounds[0])
    elif sol[0] > x_bounds[1]:
        constrs, pts, add = determine_constraint(a1, b1, a2, b2, [1, 0], x_bounds[1], x_bounds, y_bounds, constrs, pts)
        if add:
            ints.append(x_bounds[1])
    if sol[1] < y_bounds[0]:
        constrs, pts, add = determine_constraint(a1, b1, a2, b2, [0, -1], -y_bounds[0], x_bounds, y_bounds, constrs, pts)
        if add:
            ints.append(y_bounds[0])
    elif sol[1] > y_bounds[1]:
        constrs, pts, add = determine_constraint(a1, b1, a2, b2, [0, 1], y_bounds[1], x_bounds, y_bounds, constrs, pts)
        if add:
            ints.append(y_bounds[1])
    if len(ints) == 2:
        pts.append(tuple(ints))
    return constrs, pts


def point_within_bounds(sol, x_bounds, y_bounds, line_a=None):
    if line_a and len(line_a) == 2:
        if line_a[0] == 0:
            return x_bounds[0] <= sol[0] <= x_bounds[1]
        if line_a[1] == 0:
            return y_bounds[0] <= sol[1] <= y_bounds[1]
    return (x_bounds[0] <= sol[0] <= x_bounds[1]) and (y_bounds[0] <= sol[1] <= y_bounds[1])


def plot_region(ax, pt1, pt2, x_bounds, y_bounds, line_color):
    const, pts = gen_triangle_2_points(pt1[0], pt1[1], pt2[0], pt2[1], x_bounds, y_bounds)
    points = np.random.rand(len(pts), 2)
    for pos1 in range(0, len(points)):
        for pos2 in range(0, len(points[0])):
            points[pos1][pos2] = pts[pos1][pos2]
    hull = ConvexHull(points)
    ax.fill(points[hull.vertices, 0], points[hull.vertices, 1], 'lime', alpha=1)
    #ax.plot(pt1[0], pt1[1], 'bo')
    #ax.plot(pt2[0], pt2[1], 'bo')
    for pos in range(0, 3):
        c = const[pos]
        a = c[0]
        b = c[1]
        x = np.arange(0, 101)
        y = []
        for val in x:
            y.append(float(-a[0]*val + b)/a[1])
        y = np.array(y)
        ax.plot(x, y, c=line_color)
    ax.set_xlim(x_bounds[0], x_bounds[1])
    ax.set_ylim(y_bounds[0], y_bounds[1])
    return convert_constraints(const)


def gen_point_queue(num_int_points, start, end, x_bounds, y_bounds):
    queue = [start]
    for _ in range(0, num_int_points):
        if queue[len(queue) - 1][0] < (x_bounds[0] + x_bounds[1])/2:
            queue.append((np.random.randint(3*(x_bounds[0] + x_bounds[1]) / 4, x_bounds[1]),
                          np.random.randint(y_bounds[0], y_bounds[1])))
        else:
            queue.append((np.random.randint(x_bounds[0], (x_bounds[0] + x_bounds[1]) / 4),
                          np.random.randint(y_bounds[0], y_bounds[1])))
    queue.append(end)
    return queue


def convert_constraints(const):
    a_mat = []
    b_mat = []
    for c in const:
        a_mat.append(c[0])
        b_mat.append(c[1])
    return np.array(a_mat), np.array(b_mat)


if __name__ == "__main__":
    start = (10, 10)
    end = (90, 90)
    a_mats = []
    b_mats = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    q = gen_point_queue(4, start, end, (0, 100), (0, 100))
    for pos in range(0, len(q) - 1):
        a, b, = plot_region(ax, q[pos], q[pos+1], (0, 100), (0, 100), line_color="black")
        a_mats.append(a)
        b_mats.append(b)
    ax.plot(start[0], start[1], 'bo')
    ax.plot(end[0], end[1], 'bo')
    fig.show()
    print(a_mats)
    print(b_mats)
