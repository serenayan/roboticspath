import numpy as np
import matplotlib.pyplot as plt
from env_gen import load_folder_and_plot
import networkx
from networkx import Graph, all_simple_paths

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
    G = Graph()
    for k, v in dict.items():
        for v_i in v:
            G.add_edge(k, v_i)
    return G

def generate_path_list(a,b, start,end):
    G = generate_graph(a,b)
    start_node = point_in_which_regions(start, a, b)
    end_node = point_in_which_regions(end, a, b)
    assert len(start_node) > 0
    assert len(end_node) > 0
    #lets assume start and end only belong in one region for now
    #TODO: add more regions if we can handle it
    return [p for p in all_simple_paths(G, start_node[0], end_node[0])]

if __name__ == "__main__":
    s = (10, 10)
    e = (90, 90)
    x_b = (0, 100)
    y_b = (0, 100)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    a, b = load_folder_and_plot(ax, s, e, "envs/2", x_b, y_b)  # read and plot
    fig.show()
    fig = plt.figure(2)
    G = generate_graph(a,b)
    start_node = point_in_which_regions(s, a, b)
    end_node = point_in_which_regions(e, a, b)
    networkx.draw(G)
    fig.show()
    print([p for p in all_simple_paths(G, start_node[0], end_node[0])])