"""
generate_illustrative_figures.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Author: Lijun Sun.
Date: Thu Jan 7, 2021.
"""
import copy
import numpy as np

from lib.environment.matrix_world import MatrixWorld


def generate_demo_multiple_preys_pursuit_scenario():
    world_size = 40
    world_rows, world_columns = [world_size, world_size]
    global_fov_scope = 2 * world_size + 1
    problem_size = 16
    random_seed = 0

    save_path = "./data/illustration/"

    env = MatrixWorld(world_rows, world_columns,
                      n_preys=problem_size // 4,
                      n_predators=problem_size,
                      fov_scope=global_fov_scope,
                      save_path=save_path)

    frame_prefix = "MatrixWorld" + str(world_size) + \
                   "_ProblemSize" + str(problem_size) + \
                   "_RandomSeed" + str(random_seed)
    env.set_frame_prefix(frame_prefix)
    env.reset(set_seed=True, seed=random_seed)

    env.render(is_display=True,
               is_save=True, is_fixed_size=False,
               grid_on=True, tick_labels_on=False,
               show_predator_idx=False,
               show_prey_idx=False,
               show_frame_title=False)
    pass


def main():
    generate_demo_multiple_preys_pursuit_scenario()


if __name__ == "__main__":
    main()
