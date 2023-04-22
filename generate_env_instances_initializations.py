"""
generate_env_instances_initializations.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Author: Lijun SUN.
Date: Fri Dec 4, 2020.
"""
import copy
import numpy as np
import time

from lib.environment.matrix_world import MatrixWorld
from lib.biqap_solvers.permutation_initialization import \
    PermutationInitialization
from lib.fitness.fitness_multiple_prey_encircle import \
    fitness_multiple_prey_encircle
from lib.file_io.env_instances_initializations import \
    save_env_instances_initializations


def main():
    # 40, 80.
    world_size = 40
    world_rows, world_columns = [world_size, world_size]
    global_fov_scope = 2 * world_size + 1
    # 12, 2**4, 2**5, 2**6, 2**7, 2**8, 2**10
    problem_sizes = [12, 16, 20]
    random_seeds = [0, 1, 2, 3, 4]

    save_path = "./data/env_instances_initializations/"

    initializer = PermutationInitialization()

    for problem_size in problem_sizes:

        env = MatrixWorld(world_rows, world_columns,
                          n_preys=problem_size//4, n_predators=problem_size,
                          fov_scope=global_fov_scope,
                          save_path=save_path)

        for random_seed in random_seeds:

            frame_prefix = "MatrixWorld" + str(world_size) + \
                           "_ProblemSize" + str(problem_size) + \
                           "_RandomSeed" + str(random_seed) + "_Step"
            env.set_frame_prefix(frame_prefix)
            env.reset(set_seed=True, seed=random_seed)

            env.render(is_display=False,
                       is_save=True, is_fixed_size=False,
                       grid_on=True, tick_labels_on=False,
                       show_predator_idx=True,
                       show_prey_idx=True)

            prey_positions = env.get_all_preys()
            predator_positions = env.get_all_predators()

            start_time = time.time()
            greedy_initial_permutation = \
                initializer.greedy_sequential_nearest_4(
                    prey_positions, predator_positions, 1)[0]

            greedy_initial_fitness, _ = fitness_multiple_prey_encircle(
                prey_positions=prey_positions,
                predator_positions=predator_positions,
                permutation=greedy_initial_permutation)

            print("Matrix world                         :",
                  (world_rows, world_columns),
                  "\nProblem size                       :", problem_size,
                  "\nRandom seed                        :", random_seed,
                  "\nTime (s) for greedy initialization :",
                  time.time() - start_time)

            # Save.
            file_path = save_path + "MatrixWorld" + str(world_size) + ".json"
            save_env_instances_initializations(
                filename=file_path,
                problem_size=problem_size,
                random_seed=random_seed,
                prey_positions=prey_positions.tolist(),
                predator_positions=predator_positions.tolist(),
                greedy_initial_permutation=greedy_initial_permutation.tolist(),
                greedy_initial_fitness=greedy_initial_fitness)
    pass


if __name__ == "__main__":
    main()
