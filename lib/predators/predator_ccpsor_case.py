"""
predator_ccpsor_case.py
~~~~~~~~~~~~~~~~~~~~~~~

Rewrite the codes in
https://github.com/LijunSun90/pursuitCCPSOR/blob/master/lib/optimizer_cc_pso_encircle.m

AUTHOR: Lijun SUN.
Date: Fri Sep 11 2020.
~~~~~~~~~~~~~~~~~~~~~~
Modified on Fri Dec 18, 2020.
Add local capture cases checking and processing.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Modified on Tue Jan 5, 2021.
Fix bugs in updating the axial virtual robots.
"""
import copy
import numpy as np

from lib.agents.matrix_agent_global_perception import \
    MatrixAgentGlobalPerception
from lib.environment.matrix_world import MatrixWorld
from lib.fitness.fitness_single_prey_encircle import \
    fitness_single_prey_encircle


class PredatorCCPSORCase(MatrixAgentGlobalPerception):
    def __init__(self, env, idx_prey, under_debug=False):
        super(PredatorCCPSORCase, self).__init__(env, idx_prey, under_debug)

        # Algorithm parameters.
        self.w = 1
        self.c_1 = 2
        self.c_2 = 2

        # Number of iterations for no changes of the real robot.
        self.max_steps_still = 5
        self.n_steps_still = 0

        self.population_size = 20
        self.population = np.zeros((self.population_size, 2), dtype=int)
        self.population[0, :] = self.own_position.copy()

        # v1: 5. v2: 3.
        self.vicinity_size = 5
        self.vicinity_radius = int((self.vicinity_size - 1) / 2)
        self.valid_vicinity_scope = self.update_valid_vicinity_scope()

        # 5 x 5 = 25 > 20
        # 20 - 1 = 19
        self.n_virtual_robots = self.population_size - 1
        # 3 x 3 = 9
        self.n_unique_virtual_robots_T = 9
        self.virtual_robots = self.reset_virtual_robots()

        self.cluster_other_members = self.update_cluster_other_members()
        self.cluster_center_prey = self.update_cluster_center_prey()
        self.neighbor_predators, self.neighbor_preys = self.update_neighbors()

        self.velocity = np.zeros((self.population_size, 2))

        # Individual historical best of each robot.
        self.position_pi = np.zeros((self.population_size, 2))
        # Global best one in the subpopulation.
        self.position_pg = np.zeros(2)

        self.fitness = np.zeros(self.population_size)
        # Individual historical best fitness values.
        self.fitness_pi = np.zeros(self.population_size)
        # Fitness value of the global best individual in the subpopulation.
        self.fitness_pg = 0

        # The move mode among subpopulations, either sequential or parallel.
        self.subpopulation_sequential_move = False

        # Cases rules parameters.
        self.case_mask_prey = np.array([2, 2])
        self.case_mask_offset = None
        self.case_mask_other_predators = None
        self.case_mask_current_predator = None
        self.case_mask = None

        # Case masks.

        # Case 1.
        # upper.
        self.case_1 = np.array([0, 2])
        self.case_1_next_position = np.array([1, 2])
        self.case_1_free_capturing_position = np.array([1, 2])
        self.case_1_competitors = np.array([[1, 1], [1, 3]])
        # left. right. bottom.
        self.case_1_safe_other_predators = [
            np.array([[1, 1], [2, 1], [3, 1], [2, 0]]),
            np.array([[1, 3], [2, 3], [3, 3], [2, 4]]),
            np.array([[3, 1], [3, 2], [3, 3], [4, 2]])
        ]

        # Case 2.
        # left.
        self.case_2 = np.array([2, 0])
        self.case_2_next_position = np.array([2, 1])
        self.case_2_free_capturing_position = np.array([2, 1])
        self.case_2_competitors = np.array([[1, 1], [3, 1]])
        # upper.right.bottom.
        self.case_2_safe_other_predators = [
            np.array([[1, 1], [1, 2], [1, 3], [0, 2]]),
            np.array([[1, 3], [2, 3], [3, 3], [2, 4]]),
            np.array([[3, 1], [3, 2], [3, 3], [4, 2]])
        ]

        # Case 3.
        # bottom.
        self.case_3 = np.array([4, 2])
        self.case_3_next_position = np.array([3, 2])
        self.case_3_free_capturing_position = np.array([3, 2])
        self.case_3_competitors = np.array([[3, 1], [3, 3]])
        # upper.right.left.
        self.case_3_safe_other_predators = [
            np.array([[1, 1], [1, 2], [1, 3], [0, 2]]),
            np.array([[1, 3], [2, 3], [3, 3], [2, 4]]),
            np.array([[1, 1], [2, 1], [3, 1], [2, 0]])
        ]

        # Case 4.
        # right.
        self.case_4 = np.array([2, 4])
        self.case_4_next_position = np.array([2, 3])
        self.case_4_free_capturing_position = np.array([2, 3])
        self.case_4_competitors = np.array([[1, 3], [3, 3]])
        # upper.left.bottom.
        self.case_4_safe_other_predators = [
            np.array([[1, 1], [1, 2], [1, 3], [0, 2]]),
            np.array([[1, 1], [2, 1], [3, 1], [2, 0]]),
            np.array([[3, 1], [3, 2], [3, 3], [4, 2]])
        ]

        # Case 5.
        # upper left.
        self.case_5 = np.array([1, 1])
        self.case_5_next_position = np.array([2, 1])
        self.case_5_free_capturing_position = np.array([2, 1])
        self.case_5_competitors = np.array([[3, 1]])
        # upper.right.
        self.case_5_safe_other_predators = [
            np.array([[1, 2], [1, 3], [0, 2]]),
            np.array([[2, 3], [3, 3], [2, 4]])]

        # Case 6.
        # upper left.
        self.case_6 = np.array([1, 1])
        self.case_6_next_position = np.array([1, 2])
        self.case_6_free_capturing_position = np.array([1, 2])
        self.case_6_competitors = np.array([[1, 3]])
        # left.bottom.
        self.case_6_safe_other_predators = [
            np.array([[2, 1], [3, 1], [2, 0]]),
            np.array([[3, 2], [3, 3], [4, 2]])]

        # Case 7.
        # left bottom.
        self.case_7 = np.array([3, 1])
        self.case_7_next_position = np.array([3, 2])
        self.case_7_free_capturing_position = np.array([3, 2])
        self.case_7_competitors = np.array([[3, 3]])
        # upper.left.
        self.case_7_safe_other_predators = [
            np.array([[1, 2], [1, 3], [0, 2]]),
            np.array([[1, 1], [2, 1], [2, 0]])]

        # Case 8.
        # left bottom.
        self.case_8 = np.array([3, 1])
        self.case_8_next_position = np.array([2, 1])
        self.case_8_free_capturing_position = np.array([2, 1])
        self.case_8_competitors = np.array([[1, 1]])
        # bottom.right.
        self.case_8_safe_other_predators = [
            np.array([[3, 2], [3, 3], [4, 2]]),
            np.array([[1, 3], [2, 3], [2, 4]])]

        # Case 9.
        # right bottom.
        self.case_9 = np.array([3, 3])
        self.case_9_next_position = np.array([2, 3])
        self.case_9_free_capturing_position = np.array([2, 3])
        self.case_9_competitors = np.array([[1, 3]])
        # left.bottom.
        self.case_9_safe_other_predators = [
            np.array([[1, 1], [2, 1], [2, 0]]),
            np.array([[3, 1], [3, 2], [4, 2]])]

        # Case 10.
        # right bottom.
        self.case_10 = np.array([3, 3])
        self.case_10_next_position = np.array([3, 2])
        self.case_10_free_capturing_position = np.array([3, 2])
        self.case_10_competitors = np.array([[3, 1]])
        # upper.right.
        self.case_10_safe_other_predators = [
            np.array([[1, 1], [1, 2], [0, 2]]),
            np.array([[1, 3], [2, 3], [2, 4]])]

        # Case 11.
        # upper right.
        self.case_11 = np.array([1, 3])
        self.case_11_next_position = np.array([1, 2])
        self.case_11_free_capturing_position = np.array([1, 2])
        self.case_11_competitors = np.array([[1, 1]])
        # bottom.right.
        self.case_11_safe_other_predators = [
            np.array([[3, 1], [3, 2], [4, 2]]),
            np.array([[2, 3], [3, 3], [2, 4]])]

        # Case 12.
        # upper right.
        self.case_12 = np.array([1, 3])
        self.case_12_next_position = np.array([2, 3])
        self.case_12_free_capturing_position = np.array([2, 3])
        self.case_12_competitors = np.array([[3, 3]])
        # upper.left.
        self.case_12_safe_other_predators = [
            np.array([[1, 1], [1, 2], [0, 2]]),
            np.array([[2, 1], [3, 1], [2, 0]])]

        # Cases summary.
        self.cases_masks = [
            self.case_1,
            self.case_2,
            self.case_3,
            self.case_4,
            self.case_5,
            self.case_6,
            self.case_7,
            self.case_8,
            self.case_9,
            self.case_10,
            self.case_11,
            self.case_12
        ]
        self.cases_next_positions = [
            self.case_1_next_position,
            self.case_2_next_position,
            self.case_3_next_position,
            self.case_4_next_position,
            self.case_5_next_position,
            self.case_6_next_position,
            self.case_7_next_position,
            self.case_8_next_position,
            self.case_9_next_position,
            self.case_10_next_position,
            self.case_11_next_position,
            self.case_12_next_position
        ]
        self.cases_free_capturing_positions = [
            self.case_1_free_capturing_position,
            self.case_2_free_capturing_position,
            self.case_3_free_capturing_position,
            self.case_4_free_capturing_position,
            self.case_5_free_capturing_position,
            self.case_6_free_capturing_position,
            self.case_7_free_capturing_position,
            self.case_8_free_capturing_position,
            self.case_9_free_capturing_position,
            self.case_10_free_capturing_position,
            self.case_11_free_capturing_position,
            self.case_12_free_capturing_position
        ]
        self.cases_competitors = [
            self.case_1_competitors,
            self.case_2_competitors,
            self.case_3_competitors,
            self.case_4_competitors,
            self.case_5_competitors,
            self.case_6_competitors,
            self.case_7_competitors,
            self.case_8_competitors,
            self.case_9_competitors,
            self.case_10_competitors,
            self.case_11_competitors,
            self.case_12_competitors
        ]
        self.cases_safe_other_predators = [
            self.case_1_safe_other_predators,
            self.case_2_safe_other_predators,
            self.case_3_safe_other_predators,
            self.case_4_safe_other_predators,
            self.case_5_safe_other_predators,
            self.case_6_safe_other_predators,
            self.case_7_safe_other_predators,
            self.case_8_safe_other_predators,
            self.case_9_safe_other_predators,
            self.case_10_safe_other_predators,
            self.case_11_safe_other_predators,
            self.case_12_safe_other_predators
        ]
        pass

    def set_is_prey_or_not(self):
        self.is_prey = False

    def update_valid_vicinity_scope(self):
        """
        Modify self.valid_vicinity_scope.
        """
        # Get valid vicinity scope.
        # 2d numpy array of shape (2, 2):
        # [[row_min_idx, column_min_idx], [row_max_idx, column_max_idx]]
        valid_vicinity_scope = \
            self.get_valid_scope(self.own_position,
                                 self.vicinity_radius, self.vicinity_radius,
                                 self.world_rows, self.world_columns)

        self.valid_vicinity_scope = valid_vicinity_scope.copy()

        return self.valid_vicinity_scope.copy()

    def reset_virtual_robots(self):
        """
        Modify the parameters:
        self.virtual_robots, self.population
        """
        self.virtual_robots = self.random_select(self.n_virtual_robots - 4)

        # Keep the first 4 virtual robots as the axial neighbors of the current
        # real robot.
        axial_neighbors = self.own_position + self.axial_neighbors_mask
        self.virtual_robots = np.vstack((axial_neighbors, self.virtual_robots))

        self.population[1:, :] = self.virtual_robots.copy()

        return self.virtual_robots.copy()

    def random_select(self, n_select):
        """
        Random select ``n_select`` cells out of the total cells
        ``empty_cells_index``.

        :param n_select: int, >=0.
        :return: entities, where ``entities`` is a
                 (n_select, 2) numpy array, and ``empty_cells_index`` is a
                 (n - n_select, 2) numpy array.
        """
        # Get valid vicinity scope.
        # 2d numpy array of shape (2, 2):
        # [[row_min_idx, column_min_idx], [row_max_idx, column_max_idx]]
        valid_vicinity_scope = self.update_valid_vicinity_scope()

        valid_vicinity_size = \
            np.prod(valid_vicinity_scope[1, :] - valid_vicinity_scope[0, :] + 1)

        # A list of integers where each integer corresponding to some
        # kind of index.
        empty_cells_index = np.arange(valid_vicinity_size).tolist()

        replace = False
        if n_select > valid_vicinity_size:
            replace = True

        # Get coordinates of the world.
        # 0, 1, ..., (world_rows - 1); 0, 1, ..., (world_columns - 1).
        # For example,
        # array([[[0, 0, 0],
        #         [1, 1, 1],
        #         [2, 2, 2]],
        #        [[0, 1, 2],
        #         [0, 1, 2],
        #         [0, 1, 2]]])
        meshgrid_x, meshgrid_y = \
            np.mgrid[valid_vicinity_scope[0, 0]:valid_vicinity_scope[1, 0] + 1,
                     valid_vicinity_scope[0, 1]:valid_vicinity_scope[1, 1] + 1]

        # Example of meshgrid[0].flatten() is
        # array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        # Example of meshgrid[1].flatten() is
        # array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        xs, ys = meshgrid_x.flatten(), meshgrid_y.flatten()

        # Indexes.
        idx_entities = np.random.choice(empty_cells_index, n_select,
                                        replace=replace)

        # Coordinates.
        # Indexes in 2D space.
        #       |       |     |
        # 0 -idx:0--idx:1--idx:2-
        # 1 -idx:3--idx:4--idx:5-
        # 2 -idx:6--idx:7--idx:8-
        #       |       |     |
        #       0       1     2
        xs_entities, ys_entities = xs[idx_entities], ys[idx_entities]

        # Get the entities positions.
        entities = np.vstack((xs_entities, ys_entities)).T

        return entities

    @staticmethod
    def get_valid_scope(center_position, min_radius, max_radius,
                        max_row, max_column):
        """
        :param center_position: 1d numpy array of shape (2,).
        :param min_radius: int, >= 0.
        :param max_radius: int, >=0.
        :param max_row: int, > 0.
        :param max_column: int, > 0.
        :return: 2d numpy array of shape (2, 2), which is
            [[row_min_idx, column_min_idx], [row_max_idx, column_max_idx]]

        NOTE THAT both ends are included, i.e., min_idx <= idx <= max_idx.
        """
        min_idx = center_position + [-min_radius] * 2
        max_idx = center_position + [max_radius] * 2

        valid_scope = np.zeros((2, 2), dtype=np.int64)
        valid_scope[0, :] = np.maximum([0, 0], min_idx)
        valid_scope[1, :] = np.minimum([max_row, max_column], max_idx)

        return valid_scope

    def set_cluster_other_members(self, cluster_other_members):
        """
        :param cluster_other_members: 2d numpy array of shape (n_other, 2).

        Modify the parameter "self.cluster_other_members".
        """

        self.cluster_other_members = cluster_other_members.copy()

        return self.cluster_other_members.copy()

    def set_cluster_center_prey(self, cluster_center_prey):
        """
        :param cluster_center_prey: 1d numpy array of shape (2,).

        Modify the parameter "self.cluster_center_prey".
        """

        self.cluster_center_prey = cluster_center_prey.copy()

        return self.cluster_center_prey.copy()

    def set_neighbors(self, neighbor_predators, neighbor_preys):
        """
        :param neighbor_predators:
            2d numpy array of shape (n_neighbor_predators, 2).
        :param neighbor_preys:
            2d numpy array of shape (n_neighbor_preys, 2).

        Modify the parameter "self.neighbor_predators", "self.neighbor_preys".
        """

        self.neighbor_predators = neighbor_predators.copy()
        self.neighbor_preys = neighbor_preys

        return self.neighbor_predators.copy(), self.neighbor_preys.copy()

    def update_cluster_other_members(self):
        """
        Modify the parameter "self.cluster_other_members".
        """
        all_predators = \
            self.env_vectors["all_predators"].astype(np.int64).copy()
        all_predators_list = all_predators.tolist()

        all_predators_list.remove(self.own_position.tolist())

        self.cluster_other_members = np.asarray(all_predators_list)

        return self.cluster_other_members.copy()

    def update_cluster_center_prey(self):
        """
        Modify the parameter "self.cluster_center_prey".
        """
        all_preys = self.env_vectors["all_preys"].copy()

        self.cluster_center_prey = all_preys[0, :]

        return self.cluster_center_prey.copy()

    def update_neighbors(self):
        """
        Modify the parameter "self.neighbor_predators", "self.neighbor_preys".
        """
        all_predators = \
            self.env_vectors["all_predators"].astype(np.int64).copy()
        all_predators_list = all_predators.tolist()

        all_predators_list.remove(self.own_position.tolist())

        position_the_other_predators = np.asarray(all_predators_list)

        position_all_preys = self.env_vectors["all_preys"].copy()

        self.neighbor_predators = position_the_other_predators
        self.neighbor_preys = position_all_preys

        return self.neighbor_predators.copy(), self.neighbor_preys.copy()

    def update_own_position(self):
        """
        Modify the parameter "self.own_position".
        """
        self.own_position = self.env_vectors["own_position"].copy()

        self.update_valid_vicinity_scope()
        self.population[0, :] = self.own_position.copy()

    def get_action(self, sequential_move_mode=False):
        # 0. The move mode among subpopulations, either sequential or parallel.
        self.subpopulation_sequential_move = sequential_move_mode

        # 1. Perceive.
        # -> env_vectors.
        self.env_vectors = \
            self.env.perceive_globally(self.idx_agent, is_prey=False)

        # Update its related variables.
        self.update_own_position()
        self.update_neighbors()

        # 2. Update matrix representation.
        # env_vectors -> env_matrix.
        self.update_env_matrix()

        # 3. Customized part.
        # Do nothing.
        next_position = self.get_next_position()

        # Get the action.
        # 1d numpy array with the shape(2,), np.array([delta_x, delta_y]).
        direction = self.env.get_offsets(self.own_position, next_position)
        next_action = self.env.direction_action[tuple(direction)]

        # 4. Update memory.
        self.memory["env_vectors"] = copy.deepcopy(self.env_vectors)

        return next_action

    def get_next_position(self):
        # -2. Check whether the cluster center prey has already been captured.
        # If yes, there is no need to move.
        if self.is_a_prey_captured(self.cluster_center_prey.copy()):
            next_position = self.own_position.copy()

            return next_position

        # -1. Match local capture_cases.
        if not self.subpopulation_sequential_move:
            next_position = self.match_local_capture_cases()
            if next_position is not None:
                return next_position

        # 0. Update the current fitness values due to the past changes.
        # Re-evaluate the fitness value of these robots in the subpopulation.
        self.evaluate_subpopulation()

        # 1. Loop all the virtual individuals in a subpopulation.
        self.update_virtual_robots()

        # 2. Maintain the diversity of a subpopulation.
        n_unique_virtual_robots = \
            self.number_of_unique_ones(self.population[1:, :])

        if n_unique_virtual_robots <= self.n_unique_virtual_robots_T:
            self.redistribute_virtual_robots()

        # 3. Loop the real robot in the subpopulation.
        self.update_real_robot()

        next_position = self.population[0, :].copy()

        return next_position

    def match_local_capture_cases(self):
        next_position = None

        # 1. Get relative positions in the current local capture.
        self.case_mask_offset = self.cluster_center_prey - self.case_mask_prey
        self.case_mask_other_predators = \
            self.cluster_other_members - self.case_mask_offset
        self.case_mask_current_predator = \
            self.own_position - self.case_mask_offset

        # 2. Check the scope validation of the current predator.
        valid_scope_current_predator = False
        if np.max(self.case_mask_current_predator) <= 4 and \
           np.min(self.case_mask_current_predator) >= 0:
            valid_scope_current_predator = True

        if not valid_scope_current_predator:
            return next_position

        # 3. Check the scope validation of the other predators.
        row_idx_too_small = np.where(self.case_mask_other_predators < 0)[0]
        self.case_mask_other_predators = \
            np.delete(self.case_mask_other_predators, row_idx_too_small,
                      axis=0)

        row_idx_too_big = np.where(self.case_mask_other_predators > 4)[0]
        self.case_mask_other_predators = \
            np.delete(self.case_mask_other_predators, row_idx_too_big,
                      axis=0)

        valid_scope_other_predators = False
        if self.case_mask_other_predators.shape[0] > 0:
            valid_scope_other_predators = True

        if not valid_scope_other_predators:
            return next_position

        # 4. Check the local capture validation.

        # Get the current local capture case mask.
        self.case_mask = np.zeros((5, 5))
        self.case_mask[self.case_mask_other_predators[:, 0],
                       self.case_mask_other_predators[:, 1]] = 1

        # Check axial capturing positions.
        if self.is_collide(self.cluster_center_prey + [0, 1]):
            self.case_mask[2, 3] = 1

        if self.is_collide(self.cluster_center_prey + [-1, 0]):
            self.case_mask[1, 2] = 1

        if self.is_collide(self.cluster_center_prey + [0, -1]):
            self.case_mask[2, 1] = 1

        if self.is_collide(self.cluster_center_prey + [1, 0]):
            self.case_mask[3, 2] = 1

        enter_local_capture = False
        if self.case_mask.sum() >= 3:
            enter_local_capture = True

        if not enter_local_capture:
            return next_position

        # 5. Match local capturing cases.
        for idx_case in np.arange(0, 12, 1):
            next_position = self.match_one_local_capture_case(idx_case)
            if next_position is not None:
                return next_position

        return next_position

    def match_one_local_capture_case(self, idx_case):
        case = self.cases_masks[idx_case]
        case_next_position = self.cases_next_positions[idx_case]
        case_free_capturing_position = \
            self.cases_free_capturing_positions[idx_case]
        case_competitors = self.cases_competitors[idx_case]
        case_safe_other_predators = self.cases_safe_other_predators[idx_case]

        next_position = None

        # Match the case.
        satisfy_condition_1 = False
        if np.array_equal(self.case_mask_current_predator, case):
            satisfy_condition_1 = True

        satisfy_condition_2 = False
        if self.case_mask[case_free_capturing_position[0],
                          case_free_capturing_position[1]] == 0:
            satisfy_condition_2 = True

        satisfy_condition_3 = False
        for case_competitor in case_competitors:
            if case_competitor.tolist() in \
                    self.case_mask_other_predators.tolist():
                satisfy_condition_3 = True
                break

        if not (satisfy_condition_1 and satisfy_condition_2 and
                satisfy_condition_3):
            return next_position

        satisfy_condition_4 = True
        for case_safe_other_predator in case_safe_other_predators:
            if not self.case_mask[case_safe_other_predator[:, 0],
                                  case_safe_other_predator[:, 1]].any():
                satisfy_condition_4 = False
                break

        if satisfy_condition_4:
            next_position = case_next_position + self.case_mask_offset

        return next_position

    def evaluate_subpopulation(self):
        """
        Rewrite the code in
        https://github.com/LijunSun90/pursuitCCPSOR/blob/master/lib/fitness_subpopulation_restart.m

        Modify the parameters:
        self.fitness, self.fitness_pi, self.fitness_pg,
                      self.position_pi, self.position_pg
        """

        # 1. Evaluate the fitness.
        for idx in range(0, self.population_size):
            current_individual = self.population[idx, :]

            self.fitness[idx] = fitness_single_prey_encircle(
                position_prey=self.cluster_center_prey,
                use_swarm_positions=False,
                position_current=current_individual,
                position_other_predators=self.cluster_other_members,
                use_neighbors=True,
                position_neighbor_predators=self.neighbor_predators,
                position_neighbor_preys=self.neighbor_preys)

        # 2.
        # Get individual historical best and the corresponding fitness values.
        # NOTE that the individual historical best memories @position_pi and
        # @fitness_pi are not inherited in solving this dynamic optimization
        # problem.
        self.position_pi = self.population.copy()
        self.fitness_pi = self.fitness.copy()

        # 3.
        # Initialize global best individual (particle) and get the fitness_pg
        self.fitness_pg = np.min(self.fitness_pi)
        idx_min = np.argmin(self.fitness_pi)
        self.position_pg = self.position_pi[idx_min, :].copy()

    def update_virtual_robots(self):
        """
        Modify the parameters:
            self.population, self.position_pi, self.position_pg,
            self.fitness, self.fitness_pi, self.fitness_pg
        """
        for idx in range(1, self.population_size):
            # Generate the new generation.
            # PSO update equation.
            velocity_new = self.w * self.velocity[idx, :] + \
                self.c_1 * np.random.rand() * (self.position_pi[idx, :] -
                                               self.population[idx, :]) + \
                self.c_2 * np.random.rand() * (self.position_pg -
                                               self.population[idx, :])

            velocity_new = self.nearest_axial_direction(velocity_new)

            position_new = self.population[idx, :] + velocity_new

            # Restrict the position of each virtual robot to the
            # valid vicinity of the real robots.
            position_new = self.within_vicinity(position_new)

            # Always keep the first 4 virtual robots as the axial neighbors.
            if not self.subpopulation_sequential_move and idx in [1, 2, 3, 4]:
                position_new = self.own_position + \
                               self.axial_neighbors_mask[idx - 1]

            # Fitness evaluation.
            fitness_p_new = fitness_single_prey_encircle(
                position_prey=self.cluster_center_prey,
                use_swarm_positions=False,
                position_current=position_new,
                position_other_predators=self.cluster_other_members,
                use_neighbors=True,
                position_neighbor_predators=self.neighbor_predators,
                position_neighbor_preys=self.neighbor_preys)

            # Generation update.
            # Always keep the first 4 virtual robots
            if fitness_p_new <= self.fitness[idx] or (idx in [1, 2, 3, 4]):
                self.fitness[idx] = fitness_p_new
                self.velocity[idx, :] = velocity_new.copy()
                self.population[idx, :] = position_new.copy()

                # Update individual historical best.
                if fitness_p_new <= self.fitness_pi[idx]:
                    self.fitness_pi[idx] = fitness_p_new
                    self.position_pi[idx] = position_new.copy()

                # Update the global best.
                if fitness_p_new <= self.fitness_pg:
                    self.fitness_pg = fitness_p_new
                    self.position_pg = position_new.copy()

    @staticmethod
    def nearest_axial_direction(direction):
        """
        :param direction: 1d numpy array of shape (2,).
        :return: 1d numpy array of shape (2,),
            which is one of the axial directions
            {[1, 0], [0, 1], [-1, 0], [0, -1], [0, 0]}

        This is a simplified version of the Matlab code:
        https://github.com/LijunSun90/pursuitCCPSOR/blob/master/lib/one_step_direction.m
        """
        idx_abs_max = np.argmax(np.abs(direction))
        axial_direction = np.zeros(2)
        axial_direction[idx_abs_max] = np.sign(direction[idx_abs_max])

        return axial_direction.astype(int)

    def within_vicinity(self, position):
        """
        :param position: 1d numpy array of shape (2,).
        :return: 1d numpy array of shape (2,).

        Rewrite the code in
        https://github.com/LijunSun90/pursuitCCPSOR/blob/master/lib/within_scope.m
        """
        row_min_idx, column_min_idx, row_max_idx, column_max_idx = \
            self.valid_vicinity_scope.flatten()

        position[0] = np.max([row_min_idx, position[0]])
        position[0] = np.min([row_max_idx, position[0]])
        position[1] = np.max([column_min_idx, position[1]])
        position[1] = np.min([column_max_idx, position[1]])

        return position.astype(int)

    @staticmethod
    def number_of_unique_ones(swarm):
        """
        :param swarm: 2d numpy array of shape (n_individual, 2).
        :return: int,
            indicating how many unique row in swarm.
        """
        swarm_list = [tuple(row) for row in swarm]
        unique_swarm = set(swarm_list)
        n_unique_swarm = len(unique_swarm)

        return n_unique_swarm

    def redistribute_virtual_robots(self, is_evaluate=True):
        self.reset_virtual_robots()

        if is_evaluate:
            self.evaluate_subpopulation()

    def update_real_robot(self):
        """
        Modify the parameters:
        self.population, self.position_pi, self.position_pg,
        self.fitness, self.fitness_pi, self.fitness_pg
        """

        # Generate the new generation.
        velocity_new = self.position_pg - self.population[0, :]
        velocity_new = self.nearest_axial_direction(velocity_new)

        position_new = self.population[0, :] + velocity_new

        collide = self.is_collide(position_new)

        # Fitness evaluation.
        fitness_p_new = fitness_single_prey_encircle(
            position_prey=self.cluster_center_prey,
            use_swarm_positions=False,
            position_current=position_new,
            position_other_predators=self.cluster_other_members,
            use_neighbors=True,
            position_neighbor_predators=self.neighbor_predators,
            position_neighbor_preys=self.neighbor_preys)

        # Generation update.
        if not collide and fitness_p_new <= self.fitness[0]:

            self.fitness[0] = fitness_p_new
            self.velocity[0, :] = velocity_new.copy()
            self.population[0, :] = position_new.copy()

            # update.
            self.n_steps_still = 0

            # Update the global best.
            if fitness_p_new <= self.fitness_pg:
                self.fitness_pg = fitness_p_new
                self.position_pg = position_new.copy()

                # The real robot becomes the global best in the
                # subpopulation, redistribute the virtual robots around
                # the limited neighborhood.
                self.redistribute_virtual_robots(is_evaluate=False)

        # If the real robot is not the global best but keeps still,
        # then remember this.
        elif fitness_p_new > self.fitness_pg:
            self.n_steps_still += 1

        # If the real robot is not the global best and keeps still for too
        # long then update its position to help it get out of the deadlock.
        if self.subpopulation_sequential_move and \
                fitness_p_new > self.fitness_pg and \
                self.n_steps_still >= self.max_steps_still:
            # Update.
            self.n_steps_still = 0

            # Update the position.
            self.population[0, :] = self.random_walk()


def test():
    world_rows = 40
    world_columns = 40

    n_preys = 4
    n_predators = 4 * n_preys

    env = MatrixWorld(world_rows, world_columns,
                      n_preys=n_preys, n_predators=n_predators)
    env.reset(set_seed=True, seed=0)

    # 0, 16, 5, 2
    idx_predator = 0

    predator = PredatorCCPSORCase(env, idx_predator)
    print("Action :", predator.get_action())

    env.render(is_save=True)


if __name__ == "__main__":
    test()
