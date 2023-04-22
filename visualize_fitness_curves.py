"""
visualize.py
~~~~~~~~~~~~
Author: Lijun SUN.
Date: Wed Dec 16, 2020.
"""
import copy
import numpy as np
import matplotlib.pyplot as plt

from lib.file_io.env_instances_initializations import \
    load_env_instances_initializations


def fitness_curve():
    results_filename = \
        "./data/env_instances_initializations/MatrixWorld40Result.json"
    fig_save_path = "./data/env_instances_initializations/"

    env_instances = \
        load_env_instances_initializations(results_filename)

    for problem_size in env_instances.keys():
        if int(problem_size) > 12:
            continue

        if int(problem_size) == 12:
            max_iters = 333

        for instance_seed in env_instances[problem_size].keys():
            # if int(instance_seed) != 2:
            #     continue

            # Load data.
            problem = env_instances[problem_size][instance_seed]

            fitness_over_iterations_over_runs_first = \
                problem["FIRST"]["fitness_over_iterations_over_runs"][0]
            fitness_over_iterations_over_runs_best = \
                problem["BEST"]["fitness_over_iterations_over_runs"][0]
            fitness_over_iterations_over_runs_heider = \
                problem["Heider"]["fitness_over_iterations_over_runs"][0]

            n_iterations_first = len(fitness_over_iterations_over_runs_first)
            n_iterations_best = len(fitness_over_iterations_over_runs_best)
            n_iterations_heider = len(fitness_over_iterations_over_runs_heider)

            fitness_over_iterations_over_runs_heider_random = \
                problem["HeiderRandom"]["fitness_over_iterations_over_runs"]
            n_fitness_evaluations_over_runs_heider_random = \
                problem["HeiderRandom"][
                    "n_fitness_evaluations_over_runs"]
            max_fitness_evaluations_heider_random = \
                np.max(n_fitness_evaluations_over_runs_heider_random)
            avg_fitness_over_iterations_over_runs_heider_random = \
                post_processing_numerical_metric_over_steps_in_runs(
                    fitness_over_iterations_over_runs_heider_random,
                    max_fitness_evaluations_heider_random)
            n_iterations_heider_random = max_fitness_evaluations_heider_random

            fitness_over_iterations_over_runs_grasp = \
                problem["GRASP"]["fitness_over_iterations_over_runs"]
            n_iterations_over_runs_grasp = \
                [len(x) for x in fitness_over_iterations_over_runs_grasp]
            max_fitness_evaluations_grasp = max(n_iterations_over_runs_grasp)
            avg_fitness_over_iterations_over_runs_grasp = \
                post_processing_numerical_metric_over_steps_in_runs(
                    fitness_over_iterations_over_runs_grasp,
                    max_fitness_evaluations_grasp)
            n_iterations_grasp = max_fitness_evaluations_grasp

            fitness_over_iterations_over_runs_simann3 = \
                problem["SIMANN3"]["fitness_over_iterations_over_runs"]
            n_fitness_evaluations_over_runs_simann3 = \
                problem["SIMANN3"][
                    "n_fitness_evaluations_over_runs"]
            max_fitness_evaluations_simann3 = \
                np.max(n_fitness_evaluations_over_runs_simann3)
            avg_fitness_over_iterations_over_runs_simann3 = \
                post_processing_numerical_metric_over_steps_in_runs(
                    fitness_over_iterations_over_runs_simann3,
                    max_fitness_evaluations_simann3)
            n_iterations_simann3 = max_fitness_evaluations_simann3

            # Plot.
            # plt.figure()
            fig, ax = plt.subplots(constrained_layout=True)

            plt.plot(np.arange(0, n_iterations_first, 1),
                     fitness_over_iterations_over_runs_first,
                     label="FIRST")
            plt.plot(np.arange(0, n_iterations_best, 1),
                     fitness_over_iterations_over_runs_best,
                     label="BEST")
            plt.plot(np.arange(0, n_iterations_heider, 1),
                     fitness_over_iterations_over_runs_heider,
                     label="HEIDER")
            plt.plot(np.arange(0, n_iterations_grasp, 1),
                     avg_fitness_over_iterations_over_runs_grasp,
                     label="GRASP")
            plt.plot(np.arange(0, n_iterations_simann3, 1),
                     avg_fitness_over_iterations_over_runs_simann3,
                     label="SIMANN3")
            plt.plot(np.arange(0, n_iterations_heider_random, 1),
                     avg_fitness_over_iterations_over_runs_heider_random,
                     label="HeiderRandom")

            plt.plot([333] * 10, np.linspace(46.5, 57, 10), 'r--',
                     label="Time limit: 0.5s.")

            plt.legend(fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.xlabel("Iteration", fontsize=17)
            plt.ylabel("Fitness of BiQAP task allocation", fontsize=17)
            # plt.title("Problem size :" + problem_size +
            #           "; Instance :" + str(int(instance_seed) + 1))

            fig_name = fig_save_path + "Fitness_Curve_P" + problem_size + \
                "_Seed" + instance_seed + ".png"
            plt.savefig(fig_name)

            plt.show()
    pass


def post_processing_numerical_metric_over_steps_in_runs(
        metric_over_steps_in_runs, max_steps_in_runs):
    """
    :param metric_over_steps_in_runs: list of list.
    :param max_steps_in_runs: int.
    :return: a list of floats.
    """

    # 1. Complement the best_fitness_over_steps_in_runs with the same length
    # using NaN.
    metric_over_steps_in_runs = complement_to_the_same_length(
        metric_over_steps_in_runs, max_steps_in_runs)

    avg_of_runs_metric_over_steps = \
        np.nanmean(metric_over_steps_in_runs, axis=0).tolist()

    return copy.deepcopy(avg_of_runs_metric_over_steps)


def complement_to_the_same_length(a_list_of_list, max_length):

    a_list_of_list = copy.deepcopy(a_list_of_list)

    for element_list in a_list_of_list:
        current_length = len(element_list)
        length_complemented = max_length - current_length
        element_list.extend([np.nan] * length_complemented)

    return copy.deepcopy(a_list_of_list)


if __name__ == "__main__":
    fitness_curve()
