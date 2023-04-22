"""
simulated_annealing_3.py
~~~~~~~~~~~~~~~~~~~~~~~~
Reference:
[1] Burkard, Rainer E., and Eranda Çela. "Heuristics for biquadratic assignment
problems and their computational comparison." European Journal of Operational
Research 83, no. 2 (1995): 283-300.

Author: Lijun SUN.
Date: Mon Dec 10, 2020.
"""
import copy
import numpy as np


class SIMANN3:
    """
    Simulated annealing method for BiQAPs, third version.
    """
    def __init__(self, fitness_function):
        self.evaluate = fitness_function

        self.name = self.__class__.__name__

        # Algorithm parameters.

        # Control parameters.
        # r.
        self.cooling_steps = 10

        # Control parameter for the amount of considered transpositions per
        # cooling step in the phase 1, i.e., k_1 * n * (n - 1).
        self.k_1 = 0.5

        # Control parameter for the amount of considered transpositions per
        # cooling step in the phase 2, i.e., k_2 * n * (n - 1).
        self.k_2 = 0.25

        self.beta = None

        self.q = 0.8

        #
        self.initial_temperature = 2000
        self.temperature = self.initial_temperature
        self.optimal_temperature = None

        self.n_fitness_evaluations = 0

        # Performance monitor.
        self.fitness_over_iterations = []

    def reset(self):
        self.n_fitness_evaluations = 0

        self.initial_temperature = 2000
        self.temperature = self.initial_temperature
        self.optimal_temperature = None

        # Performance monitor.
        self.fitness_over_iterations = []

    def solve_an_instance(self, an_initial_solution=None):
        self.reset()

        solution_size = len(an_initial_solution)

        if solution_size == 12:
            self.initial_temperature = 2000
        elif solution_size == 16:
            self.initial_temperature = 5000
        elif solution_size == 32:
            self.initial_temperature = 20000

        permutation_solution = an_initial_solution
        fitness, _ = self.evaluate(permutation_solution)

        best_solution_so_far = permutation_solution.copy()
        best_fitness_so_far = fitness

        self.fitness_over_iterations.append(best_fitness_so_far)
        
        orders = [(i, j)
                  for i in range(solution_size)
                  for j in range(i + 1, solution_size)]

        n_total = len(orders)

        # Phase 1. Determine the optimal temperature.
        temperatures = []
        acceptance_rate_over_temperatures = []

        n_attempts_per_temperature = \
            np.floor(self.k_1 * solution_size * (solution_size - 1)).astype(int)

        for cooling_step in range(self.cooling_steps):

            temperatures.append(self.temperature)

            idx = 0
            accept_counter = 0

            for i_iter in range(n_attempts_per_temperature):

                # Search.
                i, j = orders[idx]
                tmp_permutation = self.exchange(permutation_solution, i, j)
                tmp_fitness, _ = self.evaluate(tmp_permutation)

                idx = (idx + 1) % n_total

                # Check & update.
                delta = tmp_fitness - fitness
                if delta < 0 or \
                        np.random.rand() < np.exp(-delta/self.temperature):

                    accept_counter += 1
                    permutation_solution = tmp_permutation.copy()
                    fitness = tmp_fitness

                    if fitness < best_fitness_so_far:
                        best_solution_so_far = permutation_solution.copy()
                        best_fitness_so_far = fitness

                self.fitness_over_iterations.append(best_fitness_so_far)

            # Percentage.
            acceptance_rate = accept_counter / n_attempts_per_temperature

            acceptance_rate_over_temperatures.append(acceptance_rate)

            # Cool down.
            self.temperature = self.temperature_schedule(cooling_step)

        # Get the optimal temperature.
        avg_acceptance_rate = np.mean(acceptance_rate_over_temperatures)
        deviations = np.abs(np.asarray(acceptance_rate_over_temperatures) -
                            avg_acceptance_rate)
        idx_min = np.argmin(deviations)
        self.optimal_temperature = temperatures[idx_min]

        # Phase 2.
        idx = 0

        n_attempts = \
            np.floor(self.k_2 * solution_size * (solution_size - 1)).astype(int)

        for i_iter in range(n_attempts):
            # Search.
            i, j = orders[idx]
            tmp_permutation = self.exchange(permutation_solution, i, j)
            tmp_fitness, _ = self.evaluate(tmp_permutation)

            idx = (idx + 1) % n_total

            # Check & update.
            delta = tmp_fitness - fitness
            if delta < 0 or \
                    np.random.rand() < np.exp(-delta/self.optimal_temperature):

                permutation_solution = tmp_permutation.copy()
                fitness = tmp_fitness

                if fitness < best_fitness_so_far:
                    best_solution_so_far = permutation_solution.copy()
                    best_fitness_so_far = fitness

            self.fitness_over_iterations.append(best_fitness_so_far)

        return best_solution_so_far.tolist(), best_fitness_so_far, \
            self.fitness_over_iterations.copy()

    def temperature_schedule(self, cooling_step):
        return (self.q ** cooling_step) * self.initial_temperature

    @staticmethod
    def exchange(permutation_solution, i, j):
        permutation = copy.deepcopy(permutation_solution)

        tmp = permutation[i]
        permutation[i] = permutation[j]
        permutation[j] = tmp

        return permutation


if __name__ == "__main__":
    pass
