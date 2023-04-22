"""
best_improvement_semi.py
~~~~~~~~~~~~~~~~~~~~~~~~

Reference:
[1] Burkard, Rainer E., and Eranda Çela. "Heuristics for biquadratic assignment
problems and their computational comparison." European Journal of Operational
Research 83, no. 2 (1995): 283-300.

Author: Lijun SUN.
Data: Tue Dec 8, 2020.
~~~~~~~~~~~~~~~~~~~~~~
Modified: Tue Dec 15, 2020.
Add the semi-assignment check.
"""
import copy
import numpy as np


class BEST:
    def __init__(self, fitness_function):
        self.evaluate = fitness_function

        self.name = self.__class__.__name__

        self.n_fitness_evaluations = 0

        # Performance monitor.
        self.fitness_over_iterations = []

    def reset(self):
        self.n_fitness_evaluations = 0

        # Performance monitor.
        self.fitness_over_iterations = []

    def solve_an_instance(self, an_initial_solution=None):
        self.reset()

        solution_size = len(an_initial_solution)

        permutation_solution = an_initial_solution
        fitness, _ = self.evaluate(permutation_solution)

        self.fitness_over_iterations.append(fitness)

        best_solution = permutation_solution.copy()
        best_fitness = fitness

        while True:
            is_improved = False

            for i in range(solution_size):
                for j in range(i + 1, solution_size):
                    if (i // 4) == (j // 4):
                        continue

                    tmp_permutation = self.exchange(permutation_solution, i, j)
                    tmp_fitness, _ = self.evaluate(tmp_permutation)

                    if tmp_fitness < best_fitness:
                        is_improved = True
                        best_solution = tmp_permutation
                        best_fitness = tmp_fitness

                    self.fitness_over_iterations.append(fitness)

            permutation_solution = best_solution
            fitness = best_fitness

            if is_improved is False:
                break

        return permutation_solution.tolist(), fitness, \
            self.fitness_over_iterations.copy()

    @staticmethod
    def exchange(permutation_solution, i, j):
        permutation = copy.deepcopy(permutation_solution)

        tmp = permutation[i]
        permutation[i] = permutation[j]
        permutation[j] = tmp

        return permutation


if __name__ == "__main__":
    pass
