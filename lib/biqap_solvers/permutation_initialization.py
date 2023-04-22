"""
permutation_initialization.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Author: Lijun SUN.
Date: Fri Sep 25, 2020.

Initialization for the permutation solution based population.
"""
import copy
import numpy as np

from sklearn.metrics import pairwise_distances


class PermutationInitialization:
    def __init__(self):
        pass

    def greedy_sequential_nearest_4(self,
                                    prey_positions, predators_positions,
                                    population_size):
        """
        :param prey_positions: 2d numpy array of shape (n_prey, 2).
        :param predators_positions: 2d numpy array of shape (n_predators, 2).
        :param population_size: int.
        :return: 2d numpy array of shape (population_size, solution_size),
            where a solution is a permutation.

        A solution is generated by
        1. Clustering the predators based on their relative space positions,
           and the number of clusters is n_prey.
        2. Assign each prey to its nearest unassigned cluster.

        The other solutions are generated randomly.
        """
        solution_size = len(predators_positions)

        n_prey = prey_positions.shape[0]

        # 2d numpy array of shape (n_prey, n_predators).
        pairwise_distances_prey_predators = \
            pairwise_distances(prey_positions, predators_positions)

        sorted_order_in_pairwise_distances = \
            np.argsort(pairwise_distances_prey_predators, axis=1).tolist()

        solution = []

        for idx_prey in range(n_prey):

            candidates = sorted_order_in_pairwise_distances[idx_prey][:4]
            solution.extend(candidates)

            # Update.
            for idx in range(idx_prey + 1, n_prey):
                for x in candidates:
                    sorted_order_in_pairwise_distances[idx].remove(x)

        # Solution : [2, 1, 9, 7, 8, 3, 10, 5, 4, 6, 11, 0]
        # print("Solution :", solution)

        if population_size > 1:
            left_population = self.random(population_size - 1, solution_size)
            population = np.vstack((solution, left_population))
        else:
            population = np.reshape(solution, (1, -1))

        # print("Greedy initial individual (solution) :", solution)

        return copy.deepcopy(population)

    @staticmethod
    def random(population_size, solution_size):
        """
        :param population_size: int.
        :param solution_size: int.
        :return: 2d numpy array of shape (population_size, solution_size),
            where a solution is a permutation.

        All solutions are generated randomly.
        """

        population = []
        for i_solution in range(population_size):
            solution = np.random.permutation(solution_size).astype(int)
            population.append(solution)

        population = np.asarray(population)

        return population


def test_random():
    n_prey = 3
    n_predators = 4 * n_prey

    population_size, solution_size = 10, n_predators

    initializer = PermutationInitialization()

    # prey_positions = np.random.randint(0, 30, size=(n_prey, 2))
    # predators_positions = np.random.randint(0, 30, size=(n_predators, 2))

    population = initializer.random(population_size, solution_size)

    print(population)


def test_greedy_sequential_nearest_4():
    n_prey = 3
    n_predators = 4 * n_prey
    population_size, solution_size = 10, n_predators

    initializer = PermutationInitialization()

    prey_positions = np.array([
        [2, 24],
        [24, 3],
        [6, 26]])

    predators_positions = np.array([
        [7, 1],
        [9, 29],
        [10, 23],
        [22, 0],
        [21, 28],
        [16, 1],
        [6, 5],
        [18, 17],
        [26, 1],
        [11, 15],
        [18, 8],
        [27, 19]])

    population = \
        initializer.greedy_sequential_nearest_4(prey_positions,
                                                predators_positions,
                                                population_size)

    print("Population :\n", population)


if __name__ == "__main__":
    # test_random()
    test_greedy_sequential_nearest_4()