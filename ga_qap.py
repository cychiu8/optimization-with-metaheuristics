from pickle import NONE
import random as rd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm


# hyperparameters (user inputted parameters)
PROB_CRSVR = 1
PROB_MUTATION = 0.3
K = 3  # For Tournament selection
POPULATION = 100
GENERATIONS = 30


# problems
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

# initial solution
Initial_Solution = ["D", "A", "C", "B", "G", "E", "F", "H"]
size_of_chromosome = len(Initial_Solution)
pool_of_solutions = np.empty((0, size_of_chromosome))
best_of_a_generation = np.empty((0, size_of_chromosome+1))
X0 = Initial_Solution[:]


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


def tournament_selection(solution_pools, random_size=3):
    """
    select numbers of random parents (default = 3) from the solution pool
    return the one with the best fitness value
    """
    size_of_solution_pool = len(solution_pools)
    indices_list = np.random.choice(
        size_of_solution_pool, random_size, replace=False)
    solution_mapping = {}
    for i in indices_list:
        solution_mapping[i] = objective_value(solution_pools[i])
    index_of_best_solution = min(solution_mapping, key=solution_mapping.get)
    return solution_pools[index_of_best_solution]


def find_parents_ts(all_solutions, number_of_parents=2):
    """
    find numbers of solutions from the pool as parents (default = 2)
    using the tournament selection method
    """
    # make an empty array to place the selected parents
    parents = np.empty((0, np.size(all_solutions, 1)))

    for i in range(number_of_parents):  # do the process twice to get 2 parents
        selected_parent = tournament_selection(all_solutions)

        # put the selected parent in the empty array we created above
        parents = np.vstack((parents, selected_parent))
    return parents


def is_crossover(prob_crsvr):
    """
    to check whether need to crossover
    """
    rand_num_to_crsvr_or_not = np.random.rand()
    return rand_num_to_crsvr_or_not < prob_crsvr


def crossover(parents, prob_crsvr=1):
    """
    crossover between parents to create children
    Partially- Mpped Crossover (PMX)
    """
    number_of_parent = len(parents)
    number_of_child = 2
    len_of_chromosome = len(parents[0])

    children = np.empty([number_of_child, len_of_chromosome], dtype=object)

    if is_crossover(prob_crsvr):
        cut_points = np.random.choice(len_of_chromosome, 2, replace=False)
        cut_points.sort()

        for i in range(number_of_child):
            children[i] = parents[i][:]
            reverse_map = {}
            for idx, num in enumerate(children[i]):
                reverse_map[num] = idx
            for idx in range(cut_points[0], cut_points[1]):
                target_value = parents[number_of_parent - 1 - i][idx]
                temp = children[i][idx]
                children[i][idx] = children[i][reverse_map[target_value]]
                children[i][reverse_map[target_value]] = temp

    else:
        children = parents

    return children


def is_mutation(prob_mutation):
    """
    to check whether need to mutation
    """
    rand_num_to_mutate_or_not_1 = np.random.rand()
    return rand_num_to_mutate_or_not_1 < prob_mutation


def mutation(children, prob_mutation=0.2):
    """
    mutation for children
    reverse the order
    """
    len_of_chromosome = len(children[0])
    for child in children:
        cut_points = np.random.choice(len_of_chromosome, 2, replace=False)
        cut_points.sort()
        if is_mutation(prob_mutation):
            child = np.concatenate(
                (child[:cut_points[0]], child[cut_points[0]:cut_points[1]][::-1], child[cut_points[1]:]), axis=None)

    return children


# Shuffles the elements in the vector n times and stores them
for chrome in range(POPULATION):
    new_solution = rd.sample(Initial_Solution, size_of_chromosome)
    pool_of_solutions = np.vstack((pool_of_solutions, new_solution))


for generation in tqdm(range(GENERATIONS)):
    new_population = np.empty((0, size_of_chromosome))
    new_population_with_obj_val = np.empty((0, size_of_chromosome+1))
    sorted_best_for_plotting = np.empty((0, size_of_chromosome+1))

    for family in range(int(POPULATION/2)):
        parents = find_parents_ts(pool_of_solutions)
        children = crossover(parents, PROB_CRSVR)
        mutated_children = mutation(children, PROB_MUTATION)
        for child in range(len(children)):
            mutant = np.hstack((objective_value(
                mutated_children[child]), mutated_children[child]))
        new_population = np.vstack((new_population,
                                    mutated_children))
        new_population_with_obj_val = np.vstack((new_population_with_obj_val,
                                                 mutant))

    pool_of_solutions = new_population
    sorted_best_for_plotting = np.array(sorted(new_population_with_obj_val,
                                               key=lambda x: x[0]))
    best_of_a_generation = np.vstack((best_of_a_generation,
                                      sorted_best_for_plotting[0]))


sorted_best_of_a_generation = np.array(sorted(best_of_a_generation,
                                              key=lambda x: x[0]))
best_string_overall = sorted_best_of_a_generation[0]


print()
# final solution entire chromosome
print("Final Solution (Best):", best_string_overall[1:])
final_solution_overall = objective_value(best_string_overall[1:])
print("Obj Value - Best in Generations:", round(final_solution_overall, 5))
print()
print("------------------------------")


### FOR PLOTTING THE BEST SOLUTION FROM EACH GENERATION ###

best_obj_val_overall = best_string_overall[0]


plt.plot(best_of_a_generation[:, 0])
plt.axhline(y=best_obj_val_overall, color='r', linestyle='--')

plt.title("Z Reached Through Generations", fontsize=20, fontweight='bold')
plt.xlabel("Generation", fontsize=18, fontweight='bold')
plt.ylabel("Z", fontsize=18, fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')


if sorted_best_of_a_generation[-1][0] > 2:
    k = 0.8
elif sorted_best_of_a_generation[-1][0] > 1:
    k = 0.5
elif sorted_best_of_a_generation[-1][0] > 0.5:
    k = 0.3
elif sorted_best_of_a_generation[-1][0] > 0.3:
    k = 0.2
else:
    k = 0.1


xyz2 = (GENERATIONS/6, best_obj_val_overall)
xyzz2 = (GENERATIONS/5.4, best_obj_val_overall+(k/2))

plt.annotate("Minimum Overall: %0.5f" % best_obj_val_overall, xy=xyz2, xytext=xyzz2,
             arrowprops=dict(facecolor='black', shrink=1,
                             width=1, headwidth=5),
             fontsize=12, fontweight='bold')


plt.show()
