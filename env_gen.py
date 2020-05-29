import numpy as np
import random
import matplotlib.pyplot as plt


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
    width1 = np.random.randint(10, 20) if which_small else np.random.randint(5)
    up_y1 = y1 + width1 + np.random.randint(max([-3, -width1 + 2]), 3)
    down_y1 = y1 - width1 - np.random.randint(max([-3, -width1 + 2]), 3)
    width2 = np.random.randint(5) if which_small else np.random.randint(10, 20)
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
    constraints = check_lines_against_bounds(a_up, b_up, a_down, b_down, x_bounds, y_bounds, constraints)
    constraints = check_lines_against_bounds(a_up, b_up, a_3, b_3, x_bounds, y_bounds, constraints)
    constraints = check_lines_against_bounds(a_3, b_3, a_down, b_down, x_bounds, y_bounds, constraints)
    return constraints


def determine_constraint(a1, b1, a2, b2, line_a, line_b, x_bounds, y_bounds, constraints):
    int1 = np.linalg.solve(np.array([a1, line_a]), np.array([b1, line_b]))
    int2 = np.linalg.solve(np.array([a2, line_a]), np.array([b2, line_b]))
    if point_within_bounds(int1, x_bounds, y_bounds) or point_within_bounds(int2, x_bounds, y_bounds):
        eqn = (line_a, line_b)
        if eqn not in constraints:
            constraints.append(eqn)
    return constraints


def check_lines_against_bounds(a1, b1, a2, b2, x_bounds, y_bounds, constraints):
    sol = np.linalg.solve(np.array([a1, a2]), np.array([b1, b2]))
    if sol[0] < x_bounds[0]:
        constraints = determine_constraint(a1, b1, a2, b2, [-1, 0], -x_bounds[0], x_bounds, y_bounds, constraints)
    elif sol[0] > x_bounds[1]:
        constraints = determine_constraint(a1, b1, a2, b2, [1, 0], x_bounds[1], x_bounds, y_bounds, constraints)
    if sol[1] < y_bounds[0]:
        constraints = determine_constraint(a1, b1, a2, b2, [0, -1], -y_bounds[0], x_bounds, y_bounds, constraints)
    elif sol[1] > y_bounds[1]:
        constraints = determine_constraint(a1, b1, a2, b2, [0, 1], y_bounds[1], x_bounds, y_bounds, constraints)
    return constraints


def point_within_bounds(sol, x_bounds, y_bounds):
    return (x_bounds[0] <= sol[0] <= x_bounds[1]) and (y_bounds[0] <= sol[1] <= y_bounds[1])


def plot_region(ax, pt1, pt2, x_bounds, y_bounds, line_color):
    const = gen_triangle_2_points(pt1[0], pt1[1], pt2[0], pt2[1], x_bounds, y_bounds)
    ax.plot(pt1[0], pt1[1], 'bo')
    ax.plot(pt2[0], pt2[1], 'bo')
    for pos in range(0, 3):
        c = const[pos]
        a = c[0]
        b = c[1]
        x = np.arange(0, 101)
        y = []
        for val in x:
            y.append(float(-a[0]*val + b)/a[1])
        y = np.array(y)
        plt.plot(x, y, c=line_color)
    ax.set_xlim(x_bounds[0], x_bounds[1])
    ax.set_ylim(y_bounds[0], y_bounds[1])
    return convert_constraints(const)


def gen_point_queue(num_int_points, start, end, x_bounds, y_bounds):
    queue = [start]
    for _ in range(0, num_int_points):
        if queue[len(queue) - 1][0] < (x_bounds[0] + x_bounds[1])/2:
            queue.append((np.random.randint((x_bounds[0] + x_bounds[1]) / 2, x_bounds[1]),
                          np.random.randint(y_bounds[0], y_bounds[1])))
        else:
            queue.append((np.random.randint(x_bounds[0], (x_bounds[0] + x_bounds[1]) / 2),
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
    a_mats = []
    b_mats = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    q = gen_point_queue(4, (10, 10), (90, 90), (0, 100), (0, 100))
    for pos in range(0, len(q) - 1):
        a, b, = plot_region(ax, q[pos], q[pos+1], (0, 100), (0, 100), line_color=np.random.rand(3,))
        a_mats.append(a)
        b_mats.append(b)
    fig.show()
    print(a_mats)
    print(b_mats)
