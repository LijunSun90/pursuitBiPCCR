"""
over_runs_result_of_dynamic_pursuit.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Author: Lijun SUN.
Date: Thu Nov 19, 2020.
"""
import numpy as np
import json
import os


def save_runs_result_of_dynamic_pursuit(
        filename, central_predator_name,
        world_size, n_preys,
        success_in_runs, success_rate_of_runs,
        n_steps_in_runs,
        avg_steps_of_runs, max_steps_in_runs, std_steps_of_runs,
        best_fitness_over_steps_in_runs, avg_of_runs_best_fitness_over_steps,
        n_fitness_evaluations_over_steps_in_runs,
        avg_of_runs_n_fitness_evaluations_over_steps,
        duration_time_in_runs, avg_of_runs_duration_time):

    # 1.
    if os.path.exists(filename):
        # Step 1.
        # If there is content in the file, append to it.
        file_size = os.stat(filename).st_size
    else:
        file_size = 0

    if file_size == 0:
        experimental_results = dict()
    else:
        with open(filename, 'r') as json_file:
            experimental_results = json.load(json_file)

    # 2. Write the data.
    experimental_results[central_predator_name] = dict()
    current_result = experimental_results[central_predator_name]

    current_result["world_size"] = world_size
    current_result["n_preys"] = n_preys
    current_result["success_rate_of_runs"] = success_rate_of_runs
    current_result["success_in_runs"] = success_in_runs
    current_result["n_steps_in_runs"] = n_steps_in_runs
    current_result["avg_steps_of_runs"] = avg_steps_of_runs
    current_result["max_steps_in_runs"] = max_steps_in_runs
    current_result["std_steps_of_runs"] = std_steps_of_runs
    current_result["best_fitness_over_steps_in_runs"] = \
        best_fitness_over_steps_in_runs
    current_result["avg_of_runs_best_fitness_over_steps"] = \
        avg_of_runs_best_fitness_over_steps
    current_result["n_fitness_evaluations_over_steps_in_runs"] = \
        n_fitness_evaluations_over_steps_in_runs
    current_result["avg_of_runs_n_fitness_evaluations_over_steps"] = \
        avg_of_runs_n_fitness_evaluations_over_steps
    current_result["duration_time_in_runs"] = duration_time_in_runs
    current_result["avg_of_runs_duration_time"] = avg_of_runs_duration_time

    # 3. Save the file on disk.
    with open(filename, 'w') as json_file:
        json.dump(experimental_results, json_file, indent=4)

    print("Saved to the file", filename)


if __name__ == "__main__":
    pass
