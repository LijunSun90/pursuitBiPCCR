"""
compare_biqap_solvers_over_env_instances.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Author: Lijun SUN.
Date: Mon Dec 7, 2020.
"""
import os
import numpy as np
import multiprocessing

from lib.fitness.fitness_multiple_prey_encircle import \
    fitness_multiple_prey_encircle

from lib.file_io.env_instances_initializations import \
    load_env_instances_initializations, save_env_instances_results

from lib.biqap_solvers.first_improvement_semi import FIRST
from lib.biqap_solvers.best_improvement_semi import BEST
from lib.biqap_solvers.heider_improvement_semi import Heider
from lib.biqap_solvers.heider_improvement_semi_random_order import HeiderRandom

from lib.biqap_solvers.simulated_annealing_3 import SIMANN3
from lib.biqap_solvers.greedy_randomized_adaptive_search_procedure import GRASP

prey_positions = None
predator_positions = None
greedy_initial_permutation = None
greedy_initial_fitness = None
n_fitness_evaluations = 0


def play_many_runs():
    global prey_positions, predator_positions, greedy_initial_permutation,\
           greedy_initial_fitness, n_fitness_evaluations

    # Parameters.
    # v1: 30. v2: 1.
    n_runs = 50

    # Select and uncomment a solver.
    # solver = FIRST(evaluate)
    # solver = BEST(evaluate)
    # solver = Heider(evaluate)
    # solver = HeiderRandom(evaluate)
    solver = SIMANN3(evaluate)
    # solver = GRASP(evaluate)

    # Load the test benchmark envs.
    benchmark_filename = \
        "./data/env_instances_initializations/MatrixWorld40.json"
    results_filename = \
        "./data/env_instances_initializations/MatrixWorld40Result.json"
    env_instances = load_env_instances_initializations(benchmark_filename)

    # Parse.
    problem_sizes = env_instances.keys()
    for problem_size in problem_sizes:

        if int(problem_size) > 16:
            continue

        instance_seeds = env_instances[problem_size].keys()

        for instance_seed in instance_seeds:

            # if int(instance_seed) != 0:
            #     continue

            problem = env_instances[problem_size][instance_seed]
            prey_positions = np.asarray(problem["prey_positions"])
            predator_positions = np.asarray(problem["predator_positions"])
            greedy_initial_permutation = \
                np.asarray(problem["greedy_initial_permutation"])
            greedy_initial_fitness = problem["greedy_initial_fitness"]

            n_fitness_evaluations_over_runs = []
            best_fitness_over_runs = []
            best_solution_over_runs = []
            fitness_over_iterations_over_runs = []

            for i_run in range(n_runs):
                n_fitness_evaluations = 0

                best_solution, best_fitness, fitness_over_iterations = \
                    play_once(solver)

                n_fitness_evaluations_over_runs.append(n_fitness_evaluations)
                best_fitness_over_runs.append(best_fitness)

                # best_solution_over_runs.append(best_solution.tolist())
                best_solution_over_runs.append(best_solution)

                fitness_over_iterations_over_runs.append(
                    fitness_over_iterations)

            # n_fitness_evaluations_over_runs = \
            #     np.asarray(n_fitness_evaluations_over_runs)
            # n_fitness_evaluations_over_runs *= prey_positions.shape[0]
            avg_of_runs_n_fitness_evaluations = \
                np.mean(n_fitness_evaluations_over_runs)
            std_of_runs_n_fitness_evaluations = \
                np.std(n_fitness_evaluations_over_runs)

            best_of_runs_n_fitness_evaluations = \
                np.min(n_fitness_evaluations_over_runs)
            percentage_of_runs_best_n_fitness_evaluations = \
                n_fitness_evaluations_over_runs.count(
                    best_of_runs_n_fitness_evaluations) / n_runs

            avg_of_runs_best_fitness = np.mean(best_fitness_over_runs)
            std_of_runs_best_fitness = np.std(best_fitness_over_runs)

            best_of_runs_best_fitness = np.min(best_fitness_over_runs)
            percentage_of_runs_best_fitness = \
                best_fitness_over_runs.count(best_of_runs_best_fitness) / n_runs

            save_env_instances_results(results_filename,
                                       problem_size,
                                       instance_seed,
                                       greedy_initial_permutation.tolist(),
                                       greedy_initial_fitness,
                                       solver.name,
                                       n_fitness_evaluations_over_runs,
                                       avg_of_runs_n_fitness_evaluations,
                                       std_of_runs_n_fitness_evaluations,
                                       best_fitness_over_runs,
                                       avg_of_runs_best_fitness,
                                       std_of_runs_best_fitness,
                                       best_solution_over_runs,
                                       fitness_over_iterations_over_runs)

            print("problem_size = {:4}".format(int(problem_size)),
                  ", instance_seed = {:1}".format(int(instance_seed)),
                  ", greedy_initial_fitness = {:7.3f}".format(
                      greedy_initial_fitness),
                  ", best_fitness = {:7.3f}".format(best_of_runs_best_fitness),
                  ", percentage_best_fitness = {:7.3f}".format(
                      percentage_of_runs_best_fitness),
                  ", avg_fitness = {:7.3f}".format(avg_of_runs_best_fitness),
                  ", std_fitness = {:7.3f}".format(std_of_runs_best_fitness),
                  ", best_n_fes = {:7.3f}".format(
                      best_of_runs_n_fitness_evaluations),
                  ", percentage_n_fes = {:7.3f}".format(
                      percentage_of_runs_best_n_fitness_evaluations),
                  ", avg_n_fes = {:7.3f}".format(
                      avg_of_runs_n_fitness_evaluations),
                  ", std_n_fes = {:7.3f}".format(
                      std_of_runs_n_fitness_evaluations))


def evaluate(permutation_solution=None,
             use_partial_permutation=False,
             partial_set_from_prey=None,
             partial_set_to_predator=None,
             solution_size=None,
             population=None):

    global prey_positions, predator_positions, greedy_initial_permutation, \
        greedy_initial_fitness, n_fitness_evaluations

    # ##################################################
    # Single solution evaluation.
    if population is None:

        if not use_partial_permutation:
            n_fitness_evaluations += 1
            fitness, sub_fitness_list = fitness_multiple_prey_encircle(
                prey_positions=prey_positions,
                predator_positions=predator_positions,
                permutation=permutation_solution)
        else:
            partial_size = len(partial_set_from_prey)
            n_fitness_evaluations += partial_size / solution_size

            partial_set_from_prey = \
                [partial_set_from_prey[i] // 4 for i in
                 range(0, partial_size, 4)]
            partial_prey_positions = \
                prey_positions[partial_set_from_prey, :]
            partial_predator_positions = \
                predator_positions[partial_set_to_predator, :]

            partial_size = len(partial_set_to_predator)

            fitness, sub_fitness_list = fitness_multiple_prey_encircle(
                prey_positions=partial_prey_positions,
                predator_positions=partial_predator_positions,
                permutation=np.arange(0, partial_size, 1).astype(int))

        return fitness, sub_fitness_list

    # ##################################################
    # Population evaluation.
    population_size = population.shape[0]

    n_fitness_evaluations += population_size

    if population_size > 10:
        population_preys = [prey_positions] * population_size
        population_predators = [predator_positions] * population_size

        n_processes = min(os.cpu_count(), 16)
        with multiprocessing.Pool(n_processes) as pool:
            results = \
                pool.starmap(fitness_multiple_prey_encircle,
                             zip(population_preys,
                                 population_predators,
                                 population))

        results = np.asarray(results)
        population_fitness = results[:, 0]
        population_sublist = results[:, 1].tolist()
    else:
        population_fitness = np.zeros(population_size, solution_size)
        population_sublist = [None] * population_size
        for idx in range(population_size):
            population_fitness[idx], population_sublist[idx] = \
                fitness_multiple_prey_encircle(
                    prey_positions=prey_positions,
                    predator_positions=predator_positions,
                    permutation=population[idx])

    return population_fitness, population_sublist


def play_once(solver):

    global prey_positions, predator_positions, greedy_initial_permutation

    # Normal. FIRST. BEST. Heider. HeiderRandom. GRASP.
    best_solution, best_fitness, fitness_over_iterations = \
        solver.solve_an_instance(greedy_initial_permutation)

    return best_solution, best_fitness, fitness_over_iterations


if __name__ == "__main__":
    play_many_runs()
