from env_gen import load_folder_and_plot
from obstacle_environment import ObstacleEnvironment2D
from mip_planner import MIPPlanner, TESTA, TESTB
from core.dynamics.linear_system_dynamics import LinearSystemDynamics
import matplotlib.pyplot as plt
import random
import time

def LNS(model, num_clusters, steps, time_limit=3, verbose=False):
    """Perform large neighborhood search (LNS) on model. The descent order is random.

    Arguments:
      model: the integer program.
      num_clusters: number of clusters.
      steps: the number of decompositions to apply.
      var_dict: a dict maps node index to node variable name.
      time_limit: time limit for each LNS step.
      sol: (optional) initial solution.
      graph: networkx graph object.
      verbose: if True, print objective after every decomposition.
    """

    model.max_seconds = time_limit
    total_time = 0

    var_dict = model.get_discrete_variables()

    start_time = time.time()
    sol, start_obj = initialize_solution(var_dict, model)

    cluster_list = []
    obj_list = [start_obj]

    for _ in range(steps):
        clusters = uniform_random_clusters(var_dict, num_clusters)
        # change to RL

        sol, solver_time, obj = LNS_by_clusters(
            model.copy(), clusters, var_dict, sol)
        total_time += solver_time
        if verbose:
            print("objective: ", obj)

        cur_time = time.time()

        cluster_list.append(clusters)
        obj_list.append(obj)

    return total_time, obj, start_obj - obj, cluster_list, obj_list


def LNS_by_clusters(model, clusters, var_dict, sol):
    """Perform LNS by clusters. The order of clusters is from the largest to
       the smallest.

    Arguments:
      model: the integer program.
      clusters: a dict mapping cluster index to node index.
      var_dict: mapping node index to node variable name.
      sol: current solution.

    Returns:
      new solution. time spent by the solver. new objective.
    """

    # order clusters by size from largest to smallest
    sorted_clusters = sorted(clusters.items(),
                             key=lambda x: len(x[1]),
                             reverse=True)
    solver_t = 0

    for idx, cluster in sorted_clusters:
        sol, solver_time, obj = gradient_descent(
            model.copy(), cluster, var_dict, sol)
        solver_t += solver_time

    return sol, solver_t, obj


def gradient_descent(model, cluster, var_dict, sol):
    """Perform gradient descent on model along coordinates defined by
       variables in cluster,  starting from a current solution sol.

    Arguments:
      model: the integer program.
      cluster: the coordinates to perform gradient descent on.
      var_dict: mapping node index to node variable name.
      sol: a dict representing the current solution.

    Returns:
      new_sol: the new solution.
      time: the time used to perform the descent.
      obj: the new objective value.
    """

    var_starts = []
    for k, var_list in var_dict.items():
        for v in var_list:
            # warm start variables in the current coordinate set with the existing solution.
            model_var = model.var_by_name(v)
            if k in cluster:
                var_starts.append((model_var, sol[v]))
            else:
                model += model_var == sol[v]

    # model.start = var_starts

    start_time = time.time()
    model.optimize()
    end_time = time.time()
    run_time = end_time - start_time
    new_sol = {}

    for k, var_list in var_dict.items():
        for v in var_list:
            var = model.var_by_name(v)
            try:
                new_sol[v] = round(var.x)
            except:
                return sol, run_time, -1

    return new_sol, run_time, model.objective_value

# def generate_var_dict(model):
#     '''Returns a dictionary mapping nodes in a graph to variables in a model.'''
#     model_vars = model.get_discrete_variables()
#     num_vars = 0
#     var_dict = {}

#     for model_var in model_vars:
#         if model_var.name.startswith('v'):
#             num_vars += 1

#     var_dict = dict([(i, ["v%d" %i]) for i in range(num_vars)])

#     return var_dict


def uniform_random_clusters(var_dict, num_clusters):
    '''Return a random clustering. Each node is assigned to a cluster
    a equal probability.'''

    choices = list(range(num_clusters))
    clusters = dict([(i, []) for i in range(num_clusters)])

    for k in var_dict.keys():
        cluster_choice = random.choice(choices)
        clusters[cluster_choice].append(k)

    return clusters


def initialize_solution(var_dict, model):
    '''Initialize a feasible solution.

    Arguments:
      var_dict: a dict maps node index to node variable name.

    Returns:
      a dict maps node variable name to a value.
      a torch tensor view of the dict.
    '''

    sol = {}
    # the sol_vec needs to be of type float for later use with Pytorch.
    sol_vec = None
    init_obj = 0

    for k, var_list in var_dict.items():
        #sol_vec = np.zeros((len(var_dict), len(var_list)))
        for i, v in enumerate(var_list):
            sol[v] = 0

    return sol, init_obj

START = (10, 10, 50, 0)
END = (90, 90, 0, 0)
OBSTACLES = None
x_b = (0, 100)
y_b = (0, 100)
fig = plt.figure()
ax = fig.add_subplot(111)
a, b = load_folder_and_plot(ax, (START[0], START[1]), (END[0], END[1]), "envs/1", x_b, y_b)  # read and plot
START_ENV = ObstacleEnvironment2D(START, END, OBSTACLES, a, b)
planner = MIPPlanner(START_ENV, LinearSystemDynamics(TESTA, TESTB),
                     T=200,
                     h_k=0.001)

start_time = time.time()
t, _, _, _, _ = LNS(planner, 5, 10, time_limit=1, verbose=True)
end_time = time.time()
total_t = end_time - start_time
print('solver time: ', t)
print('total time: ', total_t)

status = P.optimize(max_seconds=total_t * 10)
print(status)
print(P.objective_value)
