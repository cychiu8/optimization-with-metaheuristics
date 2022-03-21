import numpy as np
import logging


def decode_chromosome(chromosome, lower_bound, upper_bound, start_point, length):
    """
    decode the chromosome for specific variable
    """
    precision = (upper_bound - lower_bound)/((2**length)-1)
    bit_sum = 0
    for i in range(length):
        bit_sum += chromosome[start_point+i] * (2**i)
    return bit_sum * precision + lower_bound


def get_solution(chromosome):
    """
    get decoded solution from chromosome
    """
    length_chromosome_x = len(chromosome)//2
    length_chromosome_y = len(chromosome)//2
    decoded_x = decode_chromosome(chromosome=chromosome, lower_bound=-6,
                                  upper_bound=6, start_point=0, length=length_chromosome_x)
    decoded_y = decode_chromosome(chromosome=chromosome, lower_bound=-6,
                                  upper_bound=6, start_point=length_chromosome_x,
                                  length=length_chromosome_y)
    return decoded_x, decoded_y


def objective_value(chromosome):
    """
    calculate the objective value (fitness value)
    the himmelblau function
    min ((x**2)+y-11)**2+(x+(y**2)-7)**2
    """
    solution = get_solution(chromosome=chromosome)
    obj_function_value = ((solution[0]**2)+solution[1]-11)**2 + \
        (solution[0]+(solution[1]**2)-7)**2

    logging.debug("Obj Val {}".format(obj_function_value))
    return obj_function_value


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
    """
    number_of_parent = len(parents)
    number_of_child = 2
    len_of_chromosome = len(parents[0])

    children = np.empty([number_of_child, len_of_chromosome])

    if is_crossover(prob_crsvr):
        cut_points = np.random.choice(len_of_chromosome, 2, replace=False)
        cut_points.sort()

        for i in range(number_of_child):
            children[i] = np.concatenate((parents[i][:cut_points[0]],
                                          parents[number_of_parent - 1 -
                                                  i][cut_points[0]:cut_points[1]],
                                          parents[i][cut_points[1]:]), axis=None)
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
    """
    for child in children:
        for idx, gen in enumerate(child):
            if is_mutation(prob_mutation):
                if gen == 0:
                    child[idx] = 1
                else:
                    child[idx] = 0

    return children
