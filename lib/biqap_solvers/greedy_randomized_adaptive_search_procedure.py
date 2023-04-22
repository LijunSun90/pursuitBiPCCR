"""
greedy_randomized_adaptive_search_procedure.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This script implements the algorithm GRASP in the following paper.

Reference:
Mavridou, T., Panos M. Pardalos, Leonidas S. Pitsoulis, and Mauricio GC
Resende. "A GRASP for the biquadratic assignment problem." European Journal
of Operational Research 105, no. 3 (1998): 613-621.


Author: Lijun SUN.
Date: Tue Sep 1, 2020.
~~~~~~~~~~~~~~~~~~~~~~
Modified date: Wed Dec 9, 2020.
Fix bugs.
"""
import numpy as np
import random
import copy


class GRASP:
    def __init__(self, fitness_function):
        self.objective_function = fitness_function
        self.name = self.__class__.__name__

        # Algorithm parameters.
        self.RCL = None
        self.max_iteration = 1
        self.alpha = 0.25
        self.beta = 0.3
        self.lamda = 1000

        # Performance monitor.
        self.fitness_over_iterations = []

    def reset(self):
        self.RCL = None

        # Performance monitor.
        self.fitness_over_iterations = []

    def solve_an_instance(self, an_initial_solution=None):

        self.reset()

        n = len(an_initial_solution)
        initial_fitness = None
        best_solution_found = None
        best_objective_value_found = np.inf

        for i_iter in range(self.max_iteration):
            # print("="*80)
            # print("BiQAP Iteration :", i_iter)

            # Phase 1.
            solution, initial_fitness = self.construct_a_solution(n)
            # print("BiQAP : Phase 1: Construct a solution is done.")

            # Phase 2.
            solution, objective_value = self.local_search(solution)
            # print("BiQAP : Phase 2: Local search a solution is done.")

            if objective_value < best_objective_value_found:
                # print("i_iter =", i_iter)
                best_objective_value_found = objective_value
                best_solution_found = solution.copy()

            # print("BiQAP best_solution_found :", best_solution_found)
            # print("BiQAP best_objective_value_found :",
            #       best_objective_value_found)

        # print("=" * 80)
        # print("BiQAP iteration_first_found_best :",
        #       iteration_first_found_best)
        # print("BiQAP duration_time :", duration_time, "s")
        # print("=" * 80)

        return best_solution_found.tolist(), best_objective_value_found, \
            self.fitness_over_iterations.copy()

    def generate_restricted_candidate_list(self, n):
        """
        :param n: int, problem size.
        :param self.alpha: float, control parameter in deciding the RCL length.
        :param self.beta: float, control parameter in deciding the RCL length.
        :param self.lamda: int, 1 <= self.lamda <= n * n,
            maximum number of randomly generated permutation pairs.
        :return: None, however, it will
            modidy the self parameter RCL with the following form,
            [([i, j, k, m], [p, q, s, t], a_interaction_cost_value),
             ...,
             ([i, j, k, m], [p, q, s, t], a_interaction_cost_value)]

        The interaction cost of each set of four assignments must be
        computed.
        The sets of assignments with small costs are placed in a restricted
        candidate list RCL and
        one set is selected at random from the RCL.

        For reference,
        4^8 = (2^2)^8 = 2^16 = 65536.
        4^5 = (2^2)^5 = 2^10 = 1024.
        """

        self.RCL = None

        # K = {(i, j, k, m) | i, j, k, m = 1, 2, ..., n,
        #                     i != j, k, m, j != k, m, k != m}
        # |K| = n * (n - 1) * (n - 2) * (n - 3).
        # The number of feasible sets of four assignments is |K| * |K|.
        # There are O(n^8) sets of four assignments, generation of all sets
        # is time consuming.
        # Instead the algorithm generates\self.lamda random permutation pairs.
        #

        # 1. Generate self.lamda random permutation pairs.

        w_set = np.arange(start=0, stop=n, step=4)
        w_base_indexes = np.array([0, 1, 2, 3])

        # List of list with 4 elements.
        pa = []
        qa = []
        # List of floats.
        ca = []

        n_elements = 0
        while n_elements <= self.lamda:

            pa_element = \
                (np.random.choice(w_set, 1)[0] + w_base_indexes).tolist()

            qa_element = random.sample(range(0, n), 4)

            # Compute the interaction cost.
            interaction_cost = \
                self.objective_function(use_partial_permutation=True,
                                        partial_set_from_prey=pa_element,
                                        partial_set_to_predator=qa_element,
                                        solution_size=n)[0]

            pa.append(pa_element)
            qa.append(qa_element)
            ca.append(interaction_cost)

            n_elements += 1

        # 2. Sort in ascending order in terms of costs.
        ascending_order = np.argsort(ca)
        pa = [pa[i] for i in ascending_order]
        qa = [qa[i] for i in ascending_order]
        ca = [ca[i] for i in ascending_order]

        # 3. `self.beta * self.lamda` smallest elements are treated as
        # potential candidate.
        n_potential_candidates = np.floor(self.beta * self.lamda).astype(int)
        pa_potential_candidates = pa[: n_potential_candidates]
        qa_potential_candidates = qa[: n_potential_candidates]
        ca_potential_candidates = ca[: n_potential_candidates]

        # 4. `self.alpha * self.beta * self.lamda` smallest elements
        # make up of RCL.
        n_candidate = np.floor(self.alpha * self.beta * self.lamda).astype(int)
        index_candidate = random.sample(range(n_potential_candidates),
                                        n_candidate)
        # pa_candidate = [pa_potential_candidates[i] for i in index_candidate]
        # qa_candidate = [qa_potential_candidates[i] for i in index_candidate]
        # ca_candidate = [ca_potential_candidates[i] for i in index_candidate]

        pa_candidate = \
            np.asarray(pa_potential_candidates)[index_candidate].tolist()
        qa_candidate = \
            np.asarray(qa_potential_candidates)[index_candidate].tolist()
        ca_candidate = \
            np.asarray(ca_potential_candidates)[index_candidate].tolist()

        self.RCL = list(zip(pa_candidate, qa_candidate, ca_candidate))

    def construct_a_solution(self, n):

        # Stage 1.
        # Make the four initial assignments simultaneously.
        # Random select one 4-element assignments from the RCL list.

        # 0. Initially, generate the RCL if it does not exist yet.

        if self.RCL is None:
            self.generate_restricted_candidate_list(n)
            # print("BiQAP : Initial RCL has been generated.")

        # One set is selected at random from the RCL.

        n_RCL = len(self.RCL)
        index_first_4_assignments = random.sample(range(n_RCL), 1)[0]

        # list of 4 elements, list of 4 elements, a float
        partial_set_1, partial_set_2, partial_cost = \
            copy.deepcopy(self.RCL[index_first_4_assignments])

        set_a_complete_assignments = set(range(0, n))

        # Stage 2.
        # Complete the assignment with the next n-4 elements assignment.
        initial_fitness = 0
        for i in np.arange(5, n, 4):
            # One assignment is made at a time.

            # Each new assignment added to a partial permutation
            # contributes to the total cost of the assignment the cost of
            # its interaction with already made assignments.

            # To construct the permutation costs are computed
            # for each feasible assignment and among those with small cost
            # one is randomly selected to be added to the partial permutation.

            # Selects the assignment that has the minimum
            # cost of interaction with respect to the already made assignments.

            restricted_assignment_list = []
            restricted_cost_list = []

            set_already_made_assignments_from = set(partial_set_1)
            set_left_assignments_from = \
                set_a_complete_assignments - set_already_made_assignments_from
            set_left_real_assignment_from = \
                set(np.asarray(list(set_left_assignments_from)) // 4)

            set_already_made_assignments_to = set(partial_set_2)
            set_left_assignments_to = \
                set_a_complete_assignments - set_already_made_assignments_to

            real_assign_from = set_left_real_assignment_from.pop()

            assign_4_from = \
                (4 * real_assign_from + np.arange(0, 4, 1)).tolist()

            # Exhaustive search.
            for assign_to_1 in set_left_assignments_to:
                for assign_to_2 in set_left_assignments_to:

                    if assign_to_2 == assign_to_1:
                        continue

                    for assign_to_3 in set_left_assignments_to:

                        if assign_to_3 == assign_to_2 or \
                                assign_to_3 == assign_to_1:
                            continue

                        for assign_to_4 in set_left_assignments_to:

                            if assign_to_4 == assign_to_3 or \
                                    assign_to_4 == assign_to_2 or \
                                    assign_to_4 == assign_to_1:
                                continue

                            assign_4_to = [assign_to_1, assign_to_2,
                                           assign_to_3, assign_to_4]

                            tmp_partial_set_1 = partial_set_1.copy()
                            tmp_partial_set_2 = partial_set_2.copy()

                            tmp_partial_set_1.extend(assign_4_from)
                            tmp_partial_set_2.extend(assign_4_to)

                            interaction_cost = \
                                self.objective_function(
                                    use_partial_permutation=True,
                                    partial_set_from_prey=
                                    tmp_partial_set_1,
                                    partial_set_to_predator=
                                    tmp_partial_set_2,
                                    solution_size=n)[0]

                            restricted_assignment_list.append(
                                [assign_4_from, assign_4_to])
                            restricted_cost_list.append(
                                interaction_cost)

            # Sort in ascending order in terms of costs.
            ascending_order = np.argsort(restricted_cost_list)
            restricted_cost_list = \
                [restricted_cost_list[i] for i in ascending_order]
            restricted_assignment_list = \
                [restricted_assignment_list[i] for i in ascending_order]

            # `n_restricted` smallest elements are treated as potential
            # candidate.
            n_candidates = \
                (n - i + 1) / 4 * \
                (n - i + 1) * (n - i) * (n - i - 1) * (n - i - 2)

            n_restricted_list_size = \
                np.floor(self.alpha * n_candidates).astype(int)

            restricted_assignment_list = \
                restricted_assignment_list[: n_restricted_list_size]
            restricted_cost_list = \
                restricted_cost_list[: n_restricted_list_size]

            index_candidate = \
                np.random.randint(low=0, high=n_restricted_list_size, size=1)[0]

            new_assign_from, new_assign_to = \
                restricted_assignment_list[index_candidate]

            partial_set_1.extend(new_assign_from)
            partial_set_2.extend(new_assign_to)

            initial_fitness = restricted_cost_list[index_candidate]

        complete_set_1 = partial_set_1
        complete_set_2 = partial_set_2

        index = np.argsort(complete_set_1)
        solution = np.asarray([complete_set_2[i] for i in index])

        return solution, initial_fitness

    def local_search(self, solution):
        """
        2-exchange local search.
        """

        n = len(solution)
        current_objective_value = self.objective_function(solution)[0]
        current_solution = solution.copy()

        self.fitness_over_iterations.append(current_objective_value)

        is_improved = True

        while is_improved:

            is_improved = False

            # Search the neighbor.
            # First decrement search: once a better solution is found, the
            # current solution is updated and the procedure is restarted,
            # i.e., searching the new neighborhood. This is repeated until no
            # further improvement is possible.
            # Total (n * (n - 1)) / 2.
            for k in range(0, n - 1):
                for m in range(k + 1, n):
                    # Get a neighbor.
                    a_neighbor = current_solution.copy()
                    tmp = a_neighbor[k]
                    a_neighbor[k] = a_neighbor[m]
                    a_neighbor[m] = tmp

                    # Compute the cost.
                    neighbor_objective_value = \
                        self.objective_function(a_neighbor)[0]

                    # Update.
                    if neighbor_objective_value < current_objective_value:
                        is_improved = True
                        current_objective_value = neighbor_objective_value
                        current_solution = a_neighbor.copy()
                        break

                    self.fitness_over_iterations.append(current_objective_value)

                if is_improved:
                    break

            if is_improved is False:
                break

        solution = current_solution.copy()
        objective_value = current_objective_value

        return solution, objective_value


def test():
    pass


if __name__ == "__main__":
    test()
