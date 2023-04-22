"""
fitness_multiple_prey_encircle.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Author: Lijun SUN.
Date: Mon Dec 7, 2020.
"""
import numpy as np
import multiprocessing
import psutil

from .fitness_single_prey_encircle import fitness_single_prey_encircle


def fitness_multiple_prey_encircle(prey_positions, predator_positions,
                                   permutation, use_parallel=False,
                                   n_processes=4):
    """
    :param prey_positions: 2d numpy array of shape (n_preys, 2).
    :param predator_positions: 2d numpy array of shape (n_predators, 2).
    :param permutation: 1d numpy array of shape (n_predators, 2).
        The assignment between the preys and the predators.
    :param use_parallel: boolean.
    :param n_processes: int.
        The number of worker processes to use in the parallel implementation.
    :return: a float.
    """
    n_preys = prey_positions.shape[0]
    # if n_preys <= 64:
    if not use_parallel:
        fitness, sub_fitness_list = normal_implementation(
            prey_positions, predator_positions, permutation)
    else:
        fitness, sub_fitness_list = parallel_implementation(
            prey_positions, predator_positions, permutation, n_processes)

    return fitness, sub_fitness_list


def normal_implementation(prey_positions, predator_positions, permutation):
    n_preys = prey_positions.shape[0]

    sub_fitness_list = []
    for i_prey in range(n_preys):
        current_prey = prey_positions[i_prey, :]

        idx_predators = permutation[4 * i_prey + np.arange(0, 4, 1)]

        current_predators = predator_positions[idx_predators, :]

        fitness = fitness_single_prey_encircle(
            position_prey=current_prey,
            use_swarm_positions=True,
            position_predators_swarm=current_predators)

        sub_fitness_list.append(fitness)

    fitness = np.sum(sub_fitness_list)

    return fitness, sub_fitness_list


def parallel_implementation(prey_positions, predator_positions, permutation,
                            n_processes):

    solution_size = len(permutation)
    sorted_predators_positions = []
    for i in range(0, solution_size, 4):
        idx = i + np.arange(0, 4, 1)
        sorted_predators_positions.append(
            predator_positions[permutation[idx], :])

    with multiprocessing.Pool(n_processes) as pool:
        results = pool.starmap(fitness_single_prey_encircle,
                               zip(prey_positions, sorted_predators_positions))

    sub_fitness_list = results
    fitness = np.sum(sub_fitness_list)

    return fitness, sub_fitness_list
