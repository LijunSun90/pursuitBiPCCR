"""
estimate_fitness_evaluation_time.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
AUTHOR: LIJUN SUN.
DATE: THU JAN 14, 2021.
"""
import numpy as np
import time
import matplotlib.pyplot as plt

from lib.fitness.fitness_multiple_prey_encircle import \
    fitness_multiple_prey_encircle


def n_preys_to_n_predators(n):
    return 4 * n


def n_predators_to_n_preys(n):
    return n // 4


def main():
    n_runs = 50

    a_prey_position = [[1, 1]]
    a_group_predators_positions = [[1, 0], [0, 1], [1, 2], [2, 1]]

    n_preys = [1, 100, 300, 500, 700, 850, 1000, 1200]
    duration_time_over_n_preys = []

    for n in n_preys:
        preys_positions = np.asarray(a_prey_position * n)
        predators_positions = np.asarray(a_group_predators_positions * n)
        permutation = np.arange(0, 4 * n, 1)

        duration_time_over_runs = []
        for i_run in range(n_runs):

            # Normal implementation.
            start_time = time.time()
            fitness, sub_fitness_list = \
                fitness_multiple_prey_encircle(preys_positions,
                                               predators_positions,
                                               permutation,
                                               use_parallel=False)

            duration_time = time.time() - start_time

            duration_time_over_runs.append(duration_time)

        avg_duration_time = np.mean(duration_time_over_runs)
        duration_time_over_n_preys.append(avg_duration_time)

    print(duration_time_over_n_preys)

    # Plot.
    fig, ax = plt.subplots(constrained_layout=True)

    ax.plot(n_preys, duration_time_over_n_preys, '.-')
    ax.text(n_preys[0], duration_time_over_n_preys[0],
            "{:10.5f}".format(duration_time_over_n_preys[0]),
            fontsize=16, fontweight='bold')

    x = range(min(n_preys), max(n_preys))
    y = [0.5] * len(x)
    ax.plot(x, y, 'r--')

    ax.set_xlabel("No. of single-prey pursuit fitness evaluations",
                  fontsize=16)
    ax.set_ylabel("Time (s)", fontsize=16)

    ax.set_xticks(n_preys)
    ax.tick_params(labelsize=16)

    # sec_ax = ax.secondary_xaxis('top', functions=(n_preys_to_n_predators,
    #                                               n_predators_to_n_preys))
    # sec_ax.set_xlabel("BiQAP problem size")
    # n_predators = (np.asarray(n_preys) * 4).tolist()
    # sec_ax.set_xticks(n_predators)

    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
