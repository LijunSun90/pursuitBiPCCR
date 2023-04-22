"""
central_predator.py
~~~~~~~~~~~~~~~~~~~
Author: Lijun SUN.
Date: Sat Sep 12 2020.
~~~~~~~~~~~~~~~~~~~~~~
Modified: Tue 17 Nov, 2020.
1. Improve the whole codes.
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Modified: Thu 19 Nov, 2020.
1. Modify the interfaces.
"""
import os
import copy
import numpy as np
import time
import multiprocessing

from lib.fitness.fitness_multiple_prey_encircle import \
    fitness_multiple_prey_encircle


class CentralPredator:
    def __init__(self, env, biqap_solver, fitness_function=None):
        self.env = env

        if fitness_function is None:
            self.fitness_function = self.evaluate
        else:
            self.fitness_function = fitness_function

        self.biqap_solver = biqap_solver(self.fitness_function)

        self.n_predators = self.env.n_predators
        self.n_preys = self.env.n_preys

        # 2d numpy array of shape (n_predators, 2).
        self.all_predators = None
        # 2d numpy array of shape (n_preys, 2).
        self.all_preys = None

        # Memory.
        # 1d numpy array of shape (n_predators,)
        self.last_permutation_solution = None
        # Int.
        self.idx_task_allocation = 0

    def task_allocate(
            self, save_permutation_co_problem=False,
            filename=
            "./data/permutation_co_problems/permutation_co_problems.json"):

        # 0.
        start_time = time.time()

        # 1. Get the current instance.
        self.get_current_instance()

        # 2. Solve the BiQAP problem.

        permutation_solution, best_fitness, fitness_over_iterations = \
            self.biqap_solver.solve_an_instance(
                preys_positions=self.all_preys.copy(),
                predators_positions=self.all_predators.copy())

        n_fitness_evaluations = len(fitness_over_iterations)

        # 3. Update the memory.
        self.last_permutation_solution = permutation_solution.copy()
        
        self.idx_task_allocation += 1

        # 4. End the main task allocation procedures.
        end_time = time.time()
        duration_time = end_time - start_time

        # 5. Post-processing the experimental data.
        # Data transformation.
        membership_predator_to_prey_positions = \
            self.get_membership_predator_to_prey_positions(permutation_solution)

        return membership_predator_to_prey_positions, \
            permutation_solution, best_fitness, n_fitness_evaluations, \
            duration_time

    def get_current_instance(self):
        self.all_predators = self.env.get_all_predators()
        self.all_preys = self.env.get_all_preys()

        return self.all_preys.copy(), self.all_predators.copy()

    def evaluate(self, a_permutation=None, population=None):
        """
        :param a_permutation: 1d numpy array of shape (n,) or a list of int.
        :param population: 2d numpy array of shape (n, solution_size).
        :return: float.
        """

        if population is None:
            fitness, sub_fitness_list = fitness_multiple_prey_encircle(
                prey_positions=self.all_preys,
                predator_positions=self.all_predators,
                permutation=a_permutation)

            return fitness, sub_fitness_list
        else:
            population_size = population.shape[0]
            population_preys = [self.all_preys] * population_size
            population_predators = [self.all_predators] * population_size

            n_processes = min(os.cpu_count(), 16)
            with multiprocessing.Pool(n_processes) as pool:
                results = \
                    pool.starmap(fitness_multiple_prey_encircle,
                                 zip(population_preys,
                                     population_predators,
                                     population))

            results = np.asarray(results)
            population_fitness = results[:, 0]
            population_sublist = results[:, 1].tolist()

            return population_fitness, population_sublist

    @staticmethod
    def get_assignment_prey_to_predator(a_permutation=None,
                                        use_partial_permutation=False,
                                        partial_set_from_prey=None,
                                        partial_set_to_predator=None):
        """
        :param a_permutation: 1d numpy array of shape (n,),
            where each element is the assignment from prey i to predator j.
        :param use_partial_permutation: boolean.
        :param partial_set_from_prey: 1d numpy array of shape (m,), m < n.
        :param partial_set_to_predator: 1d numpy array of shape (m,), m < n.
        :return: a dict.
            assignment_prey_to_predator:
                {idx_real_prey: [idx_predator_1, ..., idx_predator_4]}

        Return the relationship between the real prey index and the
        predator index.
        """
        # Pre-processing.
        if use_partial_permutation:
            set_from_virtual_prey = partial_set_from_prey
            set_to_predator = partial_set_to_predator
        else:
            problem_size = len(a_permutation)
            set_from_virtual_prey = np.arange(0, problem_size, 1)
            set_to_predator = a_permutation

        assignment_prey_to_predator = dict()

        for idx_virtual_prey, idx_predator in \
                zip(set_from_virtual_prey, set_to_predator):

            idx_real_prey = idx_virtual_prey // 4

            if idx_real_prey in assignment_prey_to_predator.keys():
                assignment_prey_to_predator[idx_real_prey].append(idx_predator)
            else:
                assignment_prey_to_predator[idx_real_prey] = [idx_predator]

        return copy.deepcopy(assignment_prey_to_predator)

    def get_membership_predator_to_prey_positions(self, a_permutation):

        assignment_prey_to_predator = \
            self.get_assignment_prey_to_predator(a_permutation)

        membership_predator_to_prey_positions = dict()

        for idx_prey, idx_predators in assignment_prey_to_predator.items():
            for idx_predator in idx_predators:
                membership_predator_to_prey_positions[idx_predator] = dict()

                membership_predator_to_prey_positions[idx_predator]\
                    ["center_prey"] = self.all_preys[idx_prey, :]

                membership_predator_to_prey_positions[idx_predator]\
                    ["other_members"] = \
                    np.asarray([self.all_predators[idx, :]
                                for idx in idx_predators
                                if idx != idx_predator])

        return copy.deepcopy(membership_predator_to_prey_positions)


def test():
    pass


if __name__ == "__main__":
    test()
