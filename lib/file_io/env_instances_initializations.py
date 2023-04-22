"""
env_instances_initializations.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Author: Lijun SUN.
Date: Fri Dec 4, 2020.
"""
import numpy as np
import json
import os


def save_env_instances_initializations(filename,
                                       problem_size,
                                       random_seed,
                                       prey_positions,
                                       predator_positions,
                                       greedy_initial_permutation,
                                       greedy_initial_fitness):
    # Load file.
    if os.path.exists(filename):
        # Step 1.
        # If there is content in the file, append to it.
        file_size = os.stat(filename).st_size
    else:
        file_size = 0

    if file_size == 0:
        env_instances = dict()
    else:
        with open(filename, 'r') as json_file:
            env_instances = json.load(json_file)

    # Put data in place.
    if str(problem_size) in env_instances.keys():
        env_instances_problem_size = env_instances[str(problem_size)]
    else:
        env_instances[problem_size] = dict()
        env_instances_problem_size = env_instances[problem_size]

    if str(random_seed) in env_instances_problem_size.keys():
        env_instances_problem_size_random_seed = \
            env_instances_problem_size[str(random_seed)]
    else:
        env_instances_problem_size[random_seed] = dict()
        env_instances_problem_size_random_seed = \
            env_instances_problem_size[random_seed]

    env_instances_problem_size_random_seed["prey_positions"] = prey_positions
    env_instances_problem_size_random_seed["predator_positions"] = \
        predator_positions
    env_instances_problem_size_random_seed["greedy_initial_permutation"] = \
        greedy_initial_permutation
    env_instances_problem_size_random_seed["greedy_initial_fitness"] = \
        greedy_initial_fitness

    # Save file.
    with open(filename, 'w') as json_file:
        json.dump(env_instances, json_file, indent=4)


def load_env_instances_initializations(filename):
    # Read from json file.
    with open(filename, 'r') as json_file:
        env_instances_initializations = json.load(json_file)

    return env_instances_initializations


def save_env_instances_results(filename,
                               problem_size,
                               random_seed,
                               greedy_initial_permutation,
                               greedy_initial_fitness,
                               algorithm_name,
                               n_fitness_evaluations_over_runs,
                               avg_of_runs_n_fitness_evaluations,
                               std_of_runs_n_fitness_evaluations,
                               best_fitness_over_runs,
                               avg_of_runs_best_fitness,
                               std_of_runs_best_fitness,
                               best_solution_over_runs,
                               fitness_over_iterations_over_runs):
    # Load file.
    if os.path.exists(filename):
        # Step 1.
        # If there is content in the file, append to it.
        file_size = os.stat(filename).st_size
    else:
        file_size = 0

    if file_size == 0:
        env_instances = dict()
    else:
        with open(filename, 'r') as json_file:
            env_instances = json.load(json_file)

    # Put data in place.
    if str(problem_size) in env_instances.keys():
        env_instances_problem_size = env_instances[str(problem_size)]
    else:
        env_instances[problem_size] = dict()
        env_instances_problem_size = env_instances[problem_size]

    if str(random_seed) in env_instances_problem_size.keys():
        env_instances_problem_size_random_seed = \
            env_instances_problem_size[str(random_seed)]
    else:
        env_instances_problem_size[random_seed] = dict()
        env_instances_problem_size_random_seed = \
            env_instances_problem_size[random_seed]

    env_instances_problem_size_random_seed["greedy_initial_permutation"] = \
        greedy_initial_permutation
    env_instances_problem_size_random_seed["greedy_initial_fitness"] = \
        greedy_initial_fitness

    if algorithm_name in env_instances_problem_size_random_seed.keys():
        env_instance_result = \
            env_instances_problem_size_random_seed[algorithm_name]
    else:
        env_instances_problem_size_random_seed[algorithm_name] = dict()
        env_instance_result = \
            env_instances_problem_size_random_seed[algorithm_name]

    env_instance_result["n_fitness_evaluations_over_runs"] = \
        n_fitness_evaluations_over_runs
    env_instance_result["avg_of_runs_n_fitness_evaluations"] = \
        avg_of_runs_n_fitness_evaluations
    env_instance_result["std_of_runs_n_fitness_evaluations"] = \
        std_of_runs_n_fitness_evaluations

    env_instance_result["best_fitness_over_runs"] = best_fitness_over_runs
    env_instance_result["avg_of_runs_best_fitness"] = avg_of_runs_best_fitness
    env_instance_result["std_of_runs_best_fitness"] = std_of_runs_best_fitness
    env_instance_result["best_solution_over_runs"] = best_solution_over_runs
    env_instance_result["fitness_over_iterations_over_runs"] = \
        fitness_over_iterations_over_runs

    # Save file.
    with open(filename, 'w') as json_file:
        json.dump(env_instances, json_file, indent=4)

