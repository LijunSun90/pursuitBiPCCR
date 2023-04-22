"""
heider_improvement_semi_random_order.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reference:
[1] Burkard, Rainer E., and Eranda Çela. "Heuristics for biquadratic assignment
problems and their computational comparison." European Journal of Operational
Research 83, no. 2 (1995): 283-300.

Author: Lijun SUN.
Data: Tue Dec 8, 2020.
~~~~~~~~~~~~~~~~~~~~~~
Modified: Tue Dec 15, 2020.
Add the semi-assignment check.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Modified: Wed Dec 23, 2020.
Use the random order.
"""
import copy
import numpy as np

from lib.biqap_solvers.permutation_initialization import \
    PermutationInitialization


class HeiderRandom:

    def __init__(self, fitness_function):
        self.evaluate = fitness_function

        self.name = self.__class__.__name__

        self.n_fitness_evaluations = 0

        # Performance monitor.
        self.fitness_over_iterations = []

        # Algorithm parameters.
        self.initializer = PermutationInitialization()

        # Dynamic optimization parameters.
        self.last_best_solution = None

    def reset(self):
        self.n_fitness_evaluations = 0

        # Performance monitor.
        self.fitness_over_iterations = []

    def solve_an_instance(self, an_initial_solution=None,
                          preys_positions=None, predators_positions=None):

        self.reset()

        # 0. Initialization.
        if an_initial_solution is None:
            if self.last_best_solution is not None:
                an_initial_solution = self.last_best_solution
            else:
                an_initial_solution = \
                    self.initializer.greedy_sequential_nearest_4(
                        preys_positions, predators_positions, 1)[0]

        solution_size = len(an_initial_solution)

        orders = [(i, j)
                  for i in range(solution_size)
                  for j in range(i + 1, solution_size) if (i // 4) != (j // 4)]

        # Randomize the orders.
        orders = np.random.permutation(orders).tolist()

        n_total = len(orders)
        idx = 0

        permutation_solution = an_initial_solution
        fitness, _ = self.evaluate(permutation_solution)
        self.n_fitness_evaluations += 1

        self.fitness_over_iterations.append(fitness)

        while True:

            i_count = 0
            is_improved = False

            while i_count < n_total:
                i_count += 1

                i, j = orders[idx]
                tmp_permutation = self.exchange(permutation_solution, i, j)
                tmp_fitness, _ = self.evaluate(tmp_permutation)
                self.n_fitness_evaluations += 1

                if tmp_fitness < fitness:
                    is_improved = True
                    permutation_solution = tmp_permutation
                    fitness = tmp_fitness

                self.fitness_over_iterations.append(fitness)

                idx = (idx + 1) % n_total

                if is_improved:
                    break

            if is_improved is False:
                break

        # Update.
        self.last_best_solution = permutation_solution

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
