import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

Dist = pd.DataFrame([[0, 1, 2, 3, 1, 2, 3, 4], [1, 0, 1, 2, 2, 1, 2, 3], [2, 1, 0, 1, 3, 2, 1, 2],
                     [3, 2, 1, 0, 4, 3, 2, 1], [1, 2, 3, 4, 0,
                                                1, 2, 3], [2, 1, 2, 3, 1, 0, 1, 2],
                     [3, 2, 1, 2, 2, 1, 0, 1], [4, 3, 2, 1, 3, 2, 1, 0]],
                    columns=["A", "B", "C", "D", "E", "F", "G", "H"],
                    index=["A", "B", "C", "D", "E", "F", "G", "H"])

Flow = pd.DataFrame([[0, 5, 2, 4, 1, 0, 0, 6], [5, 0, 3, 0, 2, 2, 2, 0], [2, 3, 0, 0, 0, 0, 0, 5],
                     [4, 0, 0, 0, 5, 2, 2, 10], [1, 2, 0, 5, 0,
                                                 10, 0, 0], [0, 2, 0, 2, 10, 0, 5, 1],
                     [0, 2, 0, 2, 0, 5, 0, 10], [6, 0, 5, 10, 0, 1, 10, 0]],
                    columns=["A", "B", "C", "D", "E", "F", "G", "H"],
                    index=["A", "B", "C", "D", "E", "F", "G", "H"])

T0 = 1500
Temperature = T0
temp_for_plot = T0
M = 250
N = 20
alpha = 0.9

initialSolution = ["B", "D", "A", "E", "C", "F", "G", "H"]
temp = []
obj_val = []


def objective_value(array):
    """
    the objective function for qap
    """
    df = Dist.reindex(columns=array, index=array)
    arr = np.array(df)
    start = pd.DataFrame(arr*Flow)
    objarr = np.array(start)
    obj = sum(sum(objarr))
    return obj


def tempature_formula(neighbor_obj, current_obj):
    """
    the temperature formula for anneal stealing
    """
    return 1/(np.exp((neighbor_obj - current_obj)/Temperature))


def is_obj_better(neighbor_obj, current_obj):
    """
    depends on the objective function is minimize or maximize
    """
    return neighbor_obj <= current_obj


def accept_neighbor(neighbor_obj, current_obj):
    """
    whether to accept the neighbor or not
    """
    if (is_obj_better(neighbor_obj, current_obj)):
        return True
    rand_threshold = np.random.rand()
    if rand_threshold <= tempature_formula(neighbor_obj, current_obj):
        return True
    return False


def generate_neighbor(current_solution):
    """
    2-swap
    """
    ran_1 = np.random.randint(0, len(current_solution))
    ran_2 = np.random.randint(0, len(current_solution))
    while ran_1 == ran_2:
        ran_2 = np.random.randint(0, len(current_solution))
    neighbor = current_solution.copy()
    tmp = neighbor[ran_1]
    neighbor[ran_1] = neighbor[ran_2]
    neighbor[ran_2] = tmp
    return neighbor


def print_solution():
    """
    print solution
    """
    print('Solution = =', current)
    print('obj = %0.3f' % obj_val_current)


current = ["B", "D", "A", "E", "C", "F", "G", "H"]
obj_val_current = objective_value(current)

for i in range(M):

    for j in range(N):

        # search the neighbor
        neighbor = generate_neighbor(current)

        obj_val_neighbor = objective_value(neighbor)
        obj_val_current = objective_value(current)

        if (accept_neighbor(obj_val_neighbor, obj_val_current)):
            current = neighbor

    temp.append(Temperature)
    obj_val.append(obj_val_current)
    Temperature = alpha*Temperature


print_solution()

plt.plot(temp, obj_val)
plt.title("Z at Temperature Values", fontsize=20, fontweight='bold')
plt.xlabel("Temperature", fontsize=18, fontweight='bold')
plt.ylabel("Z", fontsize=18, fontweight='bold')

plt.xlim(temp_for_plot, 0)
plt.xticks(np.arange(min(temp), max(temp), 100), fontweight='bold')
plt.yticks(fontweight='bold')
plt.show()
