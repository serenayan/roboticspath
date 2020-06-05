import numpy as np
import matplotlib.pyplot as plt
from env_gen import load_folder_and_plot


def point_in_which_regions(point, a, b):
    x = point[0]
    y = point[1]
    regions = []
    for pos in range(0, len(a)):
        a_reg = a[pos]
        b_reg = b[pos]
        in_reg = True
        for reg in range(0, len(a_reg)):
            x_c = a_reg[reg][0]
            y_c = a_reg[reg][1]
            c_c = b_reg[reg]
            val = x_c*x + y_c*y
            if val > c_c:
                in_reg = False
        if in_reg:
            regions.append(pos)
    return regions


def generate_graph(a, b):
    num_regions = len(a)
    dict = {}
    for reg in range(0, num_regions):
        dict[reg] = []
    for x in range(0, 101):
        for y in range(0, 101):
            pt = (x, y)
            regs = point_in_which_regions(pt, a, b)
            for pos in range(0, len(regs) - 1):
                for pos2 in range(pos+1, len(regs)):
                    node1 = regs[pos]
                    node2 = regs[pos2]
                    if node2 not in dict[node1]:
                        dict[node1].append(node2)
                    if node1 not in dict[node2]:
                        dict[node2].append(node1)
    for reg in range(0, num_regions):
        dict[reg].sort()
    return dict


if __name__ == "__main__":
    s = (10, 10)
    e = (90, 90)
    x_b = (0, 100)
    y_b = (0, 100)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    a, b = load_folder_and_plot(ax, s, e, "envs/2", x_b, y_b)  # read and plot
    fig.show()
    print(generate_graph(a, b))