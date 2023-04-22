"""
simulate.py
~~~~~~~~~~~~
Author: Lijun SUN.
Date: Tue Nov 17, 2020.
~~~~~~~~~~~~~~~~~~~~~~~
Modified on Fri Dec 18, 2020.
Procedure of a step / iteration:
1. The preys observe, decide, and move one-by-one in a fixed order.
2. The predators observe and decide in parallel.
3. The predators move in parallel.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Modified on Sat Jan 16, 2021.
"""
import copy
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import multiprocessing

from lib.environment.matrix_world import MatrixWorld

from lib.preys.random_prey import RandomPrey

from lib.predators.central_predator import CentralPredator

from lib.predators.predator_ccpsor_case import PredatorCCPSORCase

from lib.biqap_solvers.heider_improvement_semi_random_order import HeiderRandom

from lib.file_io.over_runs_result_of_dynamic_pursuit import \
    save_runs_result_of_dynamic_pursuit


def play_once(env, biqap_solver, n_preys, n_predators, random_seed=None,
              save_frames=False, sequential_move_mode=False):

    # ==================================================
    # Parameters.
    max_steps_for_dynamic_pursuit = 1000

    # Every number of steps not to update the position of prey.
    per_steps_not_update_prey = 10

    # ==================================================
    # Performance monitor.
    best_solution_over_steps = []
    best_fitness_over_steps = []
    n_fitness_evaluations_over_steps = []
    central_predator_duration_time_over_steps = []

    # ==================================================
    # Environment.
    if random_seed is None:
        env.reset()
    else:
        env.reset(set_seed=True, seed=random_seed)

    # ==================================================
    # Agents.

    # Initialization.
    preys = initialize_prey_swarm(env)
    central_predator = initialize_central_predator(env, biqap_solver)
    predators = initialize_predator_swarm(env)

    # ==================================================
    # Play.
    success = False
    cost_steps = 0

    for step in range(max_steps_for_dynamic_pursuit):
        # print("=" * 80, "\nSTEP :", step)

        if step == 0 and save_frames:
            render_frame(env)

        # Check the capture status.
        if env.is_all_captured():
            success = True
            cost_steps = step
            print("All prey are captured. The game is over at the step " +
                  str(step) + "!")
            break

        n_collisions = 0

        # 1. The preys swarm observe, decide, and move.

        # Move in 90% time.
        if (step + 1) % per_steps_not_update_prey:
            for idx_prey in range(n_preys):
                action_prey = preys[idx_prey].get_action()
                collide = env.act(idx_prey, action_prey, is_prey=True)

                if collide:
                    n_collisions += 1

        # 2. The central predator observe and make the decision.
        membership_predator_to_prey_positions, permutation_solution, fitness, \
            n_fitness_evaluations, central_predator_duration_time = \
            central_predator.task_allocate(save_permutation_co_problem=False)
        # print("permutation_solution:", permutation_solution)

        # Task allocation.
        for idx_predator in range(n_predators):
            # Data structure: an int.
            cluster_center_prey = membership_predator_to_prey_positions\
                [idx_predator]["center_prey"]

            # Data structure: a list of integers.
            cluster_other_members = membership_predator_to_prey_positions\
                [idx_predator]["other_members"]

            predators[idx_predator].set_cluster_center_prey(cluster_center_prey)
            predators[idx_predator].set_cluster_other_members(
                cluster_other_members)

        # Determine the priorities.
        priorities = np.random.permutation(n_predators)
        # priorities = np.arange(n_predators)

        # Predaotrs observe, make decisions, and move.
        if not sequential_move_mode:
            # 3. The predators observe and make decisions in parallel.
            actions_predators = [None] * n_predators
            for idx_predator in priorities:
                action_predator = predators[idx_predator].get_action()
                actions_predators[idx_predator] = action_predator

            # Parallel implementation.
            # with multiprocessing.Pool() as pool:
            #     actions_predators = pool.map(get_a_predator_action, predators)

            # 4. The predators move in parallel.
            for idx_predator in priorities:
                action_predator = actions_predators[idx_predator]
                collide = env.act(idx_predator, action_predator, is_prey=False)

                if collide:
                    n_collisions += 1
        else:
            # 3. The predators sequentially observe, make decisions and move.
            for idx_predator in priorities:
                action_predator = predators[idx_predator].get_action(
                    sequential_move_mode=sequential_move_mode)

                collide = env.act(idx_predator, action_predator, is_prey=False)

                if collide:
                    n_collisions += 1

        if n_collisions > 0:
            print(n_collisions, "collisions in the step", step)

        if save_frames:
            render_frame(env)

        # Record the performance.
        best_solution_over_steps.append(permutation_solution)
        best_fitness_over_steps.append(fitness)
        n_fitness_evaluations_over_steps.append(n_fitness_evaluations)
        central_predator_duration_time_over_steps.append(
            central_predator_duration_time)

    if not success:
        cost_steps = max_steps_for_dynamic_pursuit

    return success, cost_steps, \
        best_solution_over_steps, best_fitness_over_steps, \
        n_fitness_evaluations_over_steps,\
        central_predator_duration_time_over_steps


def get_a_predator_action(predator):

    return predator.get_action()


def play_many_runs():

    # ==================================================
    # Parameters.
    # v1: False. v2: True.
    save_frames = True
    sequential_move_mode = False

    # v1: 50. v2: 1.
    n_runs = 1

    # Random seed 0 is the Instance 1 in the paper.
    a_random_seed = 0
    # random_seeds = np.arange(0, n_runs, 1)
    random_seeds = [a_random_seed] * n_runs

    world_size = 40

    # v1: 3. v2: 4.
    n_preys = 3
    n_predators = 4 * n_preys

    biqap_solver = HeiderRandom

    filename_move_mode = "_Parallel"
    if sequential_move_mode:
        filename_move_mode = "_Sequential"

    filename = "Multiple_Preys_Pursuit_WorldSize" + str(world_size) + \
               "_Preys" + str(n_preys) + "_Seed" + str(a_random_seed) + \
               filename_move_mode + ".json"
    filename = os.path.join("data/multiple_preys_pursuit/", filename)

    # ==================================================
    # Performance monitor.
    success_in_runs = []
    n_steps_in_runs = []

    best_solution_over_steps_in_runs = []
    best_fitness_over_steps_in_runs = []
    n_fitness_evaluations_over_steps_in_runs = []
    central_predator_duration_time_over_steps_in_runs = []
    duration_time_in_runs = []

    # ==================================================
    # Prepare the environment.
    env = create_environment(world_size, n_preys, n_predators)

    # ==================================================
    # Run.
    print("filename :", filename)
    for i_run in range(n_runs):
        print("="*80)
        print("i_run :", i_run, "...")

        random_seed = random_seeds[i_run]

        start_time = time.time()

        success_or_not, cost_steps, \
            best_solution_over_steps, best_fitness_over_steps, \
            n_fitness_evaluations_over_steps, \
            central_predator_duration_time_over_steps = \
            play_once(env, biqap_solver, n_preys, n_predators,
                      random_seed=random_seed, save_frames=save_frames,
                      sequential_move_mode=sequential_move_mode)

        duration_time = time.time() - start_time
        print("Run", i_run, " time (s):", duration_time)

        success_in_runs.append(success_or_not)
        n_steps_in_runs.append(cost_steps)
        best_solution_over_steps_in_runs.append(best_solution_over_steps)
        best_fitness_over_steps_in_runs.append(best_fitness_over_steps)
        n_fitness_evaluations_over_steps_in_runs.append(
            n_fitness_evaluations_over_steps)
        central_predator_duration_time_over_steps_in_runs.append(
            central_predator_duration_time_over_steps)
        duration_time_in_runs.append(duration_time)

    # ==================================================
    # Post-processing.
    post_processing_visualize_save(
        filename,
        biqap_solver.__class__.__name__,
        world_size,
        n_preys,
        success_in_runs,
        n_steps_in_runs,
        best_fitness_over_steps_in_runs,
        n_fitness_evaluations_over_steps_in_runs,
        central_predator_duration_time_over_steps_in_runs,
        duration_time_in_runs)


def create_environment(world_size, n_preys, n_predators):

    # Configurations.
    world_rows = world_size
    world_columns = world_size
    global_fov_scope = 2 * world_size + 1

    env = MatrixWorld(world_rows, world_columns,
                      n_preys=n_preys, n_predators=n_predators,
                      fov_scope=global_fov_scope)

    return env


def render_frame(env):

    # Modify `is_display`.
    env.render(is_display=True,
               is_save=True, is_fixed_size=False,
               grid_on=True, tick_labels_on=False,
               show_predator_idx=False,
               show_prey_idx=False)


def initialize_prey_swarm(env):
    """
    :param env:
    :return: A list of prey,
             which are some kind of prey class instance.
    """
    n_preys = env.n_preys

    # [8]
    # np.arange(0, n_prey, 1).tolist()
    debug_idx =[]

    preys = []
    for idx in range(n_preys):
        under_debug = False
        if idx in debug_idx:
            under_debug = True

        prey = RandomPrey(env, idx, under_debug=under_debug)

        preys.append(prey)

    return preys


def initialize_predator_swarm(env):
    """
    :param env:
    :return: A list of predators,
             which are some kind of predator class instance.
    """

    n_predators = env.n_predators

    # [8]
    # np.arange(0, n_predators, 1).tolist()
    debug_idx =[]

    predators = []
    for idx in range(n_predators):
        under_debug = False
        if idx in debug_idx:
            under_debug = True

        # predator = DoNothingPredator(env, idx, under_debug=under_debug)
        predator = PredatorCCPSORCase(env, idx, under_debug=under_debug)

        predators.append(predator)

    return predators


def initialize_central_predator(env, biqap_solver):
    """
    :param env:
    :return: A central predators,
             which is some kind of central predator class instance.
    """

    central_predator = CentralPredator(env, biqap_solver)

    return central_predator


def post_processing_visualize_save(
        filename,
        biqap_solver_name,
        world_size, n_preys,
        success_in_runs,
        n_steps_in_runs,
        best_fitness_over_steps_in_runs,
        n_fitness_evaluations_over_steps_in_runs,
        central_predator_duration_time_over_steps_in_runs,
        duration_time_in_runs):

    # Parameters.
    n_runs = len(success_in_runs)

    # ==================================================
    # Post-processing the experimental results.

    success_rate_of_runs = np.sum(success_in_runs) / n_runs
    avg_steps_of_runs = np.mean(n_steps_in_runs)
    max_steps_in_runs = int(np.max(n_steps_in_runs))
    std_steps_of_runs = np.std(n_steps_in_runs)

    avg_of_runs_best_fitness_over_steps = \
        post_processing_numerical_metric_over_steps_in_runs(
            best_fitness_over_steps_in_runs, max_steps_in_runs)

    avg_of_runs_n_fitness_evaluations_over_steps = \
        post_processing_numerical_metric_over_steps_in_runs(
            n_fitness_evaluations_over_steps_in_runs,
            max_steps_in_runs)

    avg_of_runs_central_predator_duration_time_over_steps = \
        post_processing_numerical_metric_over_steps_in_runs(
            central_predator_duration_time_over_steps_in_runs,
            max_steps_in_runs)
    avg_central_predator_duration_time_per_steps = \
        np.mean(avg_of_runs_central_predator_duration_time_over_steps)
    avg_of_runs_duration_time = np.mean(duration_time_in_runs)

    # Visualize.
    print("=" * 80)
    print("World size :", world_size, "; Preys :", n_preys)
    print("Success rate :", success_rate_of_runs)
    print("Avg steps :", avg_steps_of_runs)
    print("Std steps :", std_steps_of_runs)
    print("Avg central predator time (s) per step :",
          avg_central_predator_duration_time_per_steps)
    print("Avg duration time (s) per run :", avg_of_runs_duration_time)

    visualize(max_steps_in_runs,
              best_fitness_over_steps_in_runs,
              avg_of_runs_best_fitness_over_steps,
              n_fitness_evaluations_over_steps_in_runs,
              avg_of_runs_n_fitness_evaluations_over_steps,
              avg_of_runs_central_predator_duration_time_over_steps)

    # Save.
    save_runs_result_of_dynamic_pursuit(
        filename,
        biqap_solver_name,
        world_size, n_preys,
        success_in_runs, success_rate_of_runs,
        n_steps_in_runs,
        avg_steps_of_runs, max_steps_in_runs, std_steps_of_runs,
        best_fitness_over_steps_in_runs, avg_of_runs_best_fitness_over_steps,
        n_fitness_evaluations_over_steps_in_runs,
        avg_of_runs_n_fitness_evaluations_over_steps,
        duration_time_in_runs, avg_of_runs_duration_time)


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


def visualize(max_steps_in_runs,
              best_fitness_over_steps_in_runs,
              avg_of_runs_best_fitness_over_steps,
              n_fitness_evaluations_over_steps_in_runs,
              avg_of_runs_n_fitness_evaluations_over_steps,
              avg_of_runs_central_predator_duration_time_over_steps):

    # ==================================================
    # Best fitness over steps.
    plt.figure()

    filtered_data = pre_processing_visualized_data(
        best_fitness_over_steps_in_runs, max_steps_in_runs)

    for idx_row, row in enumerate(filtered_data):

        if idx_row % 20 != 0:
            continue

        n_runs = len(row)
        plt.scatter([idx_row + 1] * n_runs, row)

    plt.plot(np.arange(1, max_steps_in_runs + 1),
             avg_of_runs_best_fitness_over_steps, 'r--', label="Average curve")

    plt.legend()
    plt.xlabel("Step")
    plt.ylabel("BiQAP multiple-prey pursuit fitness")

    # ==================================================
    # n fitness evaluation over steps.
    plt.figure()

    filtered_data = pre_processing_visualized_data(
        n_fitness_evaluations_over_steps_in_runs,
        max_steps_in_runs)

    for idx_row, row in enumerate(filtered_data):

        if idx_row % 20 != 0:
            continue

        n_runs = len(row)
        plt.scatter([idx_row + 1] * n_runs, row)

    plt.plot(np.arange(1, max_steps_in_runs + 1),
             avg_of_runs_n_fitness_evaluations_over_steps, 'r--',
             label="Average curve")

    plt.legend()
    plt.xlabel("Step")
    plt.ylabel("No. of multiple-prey pursuit fitness evaluations")

    plt.show()


def pre_processing_visualized_data(a_list_of_list, max_length):

    a_list_of_list = complement_to_the_same_length(a_list_of_list, max_length)

    data = np.asarray(a_list_of_list).T
    mask = ~np.isnan(data)
    filtered_data = [d[m] for d, m in zip(data, mask)]

    return copy.deepcopy(filtered_data)


if __name__ == "__main__":
    play_many_runs()
