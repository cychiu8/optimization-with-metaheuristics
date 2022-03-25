import time
import random as rd
import numpy as np
import matplotlib.pyplot as plt
import ga_functions_continuous as genf
from tqdm import tqdm


# hyperparameters (user inputted parameters)
PROB_CRSVR = 1
PROB_MUTATION = 0.3
POPULATION = 120
GENERATIONS = 80


# enconding decision variables (x, y)
# initial solution
x_y_string = np.array([0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1,
                       0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1])
size_of_chromosome = len(x_y_string)
pool_of_solutions = np.empty((0, size_of_chromosome))
best_of_a_generation = np.empty((0, size_of_chromosome+1))

for chrome in range(POPULATION):
    rd.shuffle(x_y_string)
    pool_of_solutions = np.vstack((pool_of_solutions, x_y_string))

start_time = time.time()
for generation in tqdm(range(GENERATIONS)):
    new_population = np.empty((0, size_of_chromosome))
    new_population_with_obj_val = np.empty((0, size_of_chromosome+1))
    sorted_best_for_plotting = np.empty((0, size_of_chromosome+1))

    for family in range(int(POPULATION/2)):

        parents = genf.find_parents_ts(pool_of_solutions)
        children = genf.crossover(parents, PROB_CRSVR)
        mutated_children = genf.mutation(children, PROB_MUTATION)

        for child in range(len(children)):
            mutant = np.hstack((genf.objective_value(
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


end_time = time.time()


sorted_best_of_a_generation = np.array(sorted(best_of_a_generation,
                                              key=lambda x: x[0]))
best_string_overall = sorted_best_of_a_generation[0]

print("------------------------------")
print("Execution Time in Seconds:", end_time - start_time)  # exec. time

print()
# final solution entire chromosome
print("Final Solution (Best):", best_string_overall[1:])
# final solution x chromosome
print("Encoded X (Best):", best_string_overall[1:14])
# final solution y chromosome
print("Encoded Y (Best):", best_string_overall[14:])

final_solution_overall = genf.objective_value(best_string_overall[1:])

print("Decoded X (Best):", round(
    genf.get_solution(best_string_overall[1:])[0], 5))  # real value of x
print("Decoded Y (Best):", round(
    genf.get_solution(best_string_overall[1:])[0], 5))  # real value of y
# obj val of final chromosome
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
