"""
matrix_agent_global_perception.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AUTHOR: Lijun SUN.
Date: Fri Sep 11 2020.
~~~~~~~~~~~~~~~~~~~~~~~~~
Modified: Wed 17 Nov 2020.
1. Fix the bug in the random walk in avoiding returning to the last position.
"""
import copy
import numpy as np
from abc import ABCMeta, abstractmethod


# Base class.
class MatrixAgentGlobalPerception(metaclass=ABCMeta):
    def __init__(self, env, idx_agent, under_debug=False):
        self.env = env
        self.idx_agent = idx_agent
        self.under_debug = under_debug

        # Get environment parameters.
        self.world_rows = self.env.world_rows
        self.world_columns = self.env.world_columns

        # int.
        self.fov_scope = self.env.fov_scope
        # 1d numpy array of shape (2,).
        self.fov_offsets_in_padded = self.env.fov_offsets_in_padded.copy()
        # 2d numpy array of shape (2, 2).
        self.fov_mask_in_padded = self.env.fov_mask_in_padded.copy()
        # 2d numpy array of shape (2, 2).
        self.fov_global_scope_in_padded = \
            self.env.fov_global_scope_in_padded.copy()

        # 2d numpy array of shape (4, 2).
        self.axial_neighbors_mask = self.env.axial_neighbors_mask

        # 2d numpy array of shape (8, 2).
        self.two_steps_away_neighbors_mask = \
            self.env.two_steps_away_neighbors_mask

        # Own parameters.
        self.is_prey = False
        self.env_vectors = None
        self.env_matrix = None
        self.padded_env_matrix = None

        self.own_position = None

        self.memory = None

        self.set_is_prey_or_not()
        self.reset()

    @abstractmethod
    def set_is_prey_or_not(self):
        self.is_prey = False

    def reset(self):
        # 0.
        self.memory = dict()

        # 1. Perceive.
        # -> env_vectors.
        self.env_vectors = \
            self.env.perceive_globally(self.idx_agent, is_prey=self.is_prey)
        self.own_position = self.env_vectors["own_position"].copy()

        # 2. Update matrix representation.
        # env_vectors -> env_matrix.
        self.update_env_matrix()

        # 3. Update memory.
        self.memory["env_vectors"] = copy.deepcopy(self.env_vectors)

    def update_env_matrix(self):
        """
        env_vectors -> env_matrix.
        """
        # 1. Parse.
        all_preys = self.env_vectors["all_preys"]
        all_predators = self.env_vectors["all_predators"].astype(np.int64)
        found_obstacles = self.env_vectors["found_obstacles"]

        # 2. Build / Update.
        if self.padded_env_matrix is None:
            self.padded_env_matrix = \
                self.env.create_padded_env_matrix_from_vectors(
                    self.world_rows, self.world_columns, self.fov_scope,
                    all_preys, all_predators, found_obstacles)

            self.env_matrix = self.padded_env_matrix[
                              self.fov_global_scope_in_padded[0, 0]:
                              self.fov_global_scope_in_padded[1, 0],
                              self.fov_global_scope_in_padded[0, 1]:
                              self.fov_global_scope_in_padded[1, 1], :]
        else:
            self.update_preys_matrix(all_preys)
            self.update_predators_matrix(all_predators)
            self.update_obstacles_matrix(found_obstacles)

    def update_preys_matrix(self, new_positions):
        """
        :param new_positions: 2d numpy array of shape (x, 2) where 0 <= x.
        :return: None.
        """
        old_positions = self.memory["env_vectors"]["all_preys"]

        self.env_matrix[old_positions[:, 0], old_positions[:, 1], 0] = 0
        self.env_matrix[new_positions[:, 0], new_positions[:, 1], 0] = 1

        old_positions_in_padded = old_positions + self.fov_offsets_in_padded
        new_positions_in_padded = new_positions + self.fov_offsets_in_padded

        self.padded_env_matrix[old_positions_in_padded[:, 0],
                               old_positions_in_padded[:, 1], 0] = 0
        self.padded_env_matrix[new_positions_in_padded[:, 0],
                               new_positions_in_padded[:, 1], 0] = 1

    def update_predators_matrix(self, new_positions):
        """
        :param new_positions: 2d numpy array of shape (x, 2) where 0 <= x.
        :return: None.
        """
        # 1. Update unknown map.
        for new_position in new_positions:
            fov_idx = self.fov_mask_in_padded + new_position
            self.padded_env_matrix[fov_idx[0, 0]: fov_idx[1, 0],
                                   fov_idx[0, 1]: fov_idx[1, 1], 3] = 0

        # 2. Update predator channel.
        old_positions = \
            self.memory["env_vectors"]["all_predators"].astype(np.int64)

        self.env_matrix[old_positions[:, 0], old_positions[:, 1], 1] = 0
        self.env_matrix[new_positions[:, 0], new_positions[:, 1], 1] = 1

        old_positions_in_padded = old_positions + self.fov_offsets_in_padded
        new_positions_in_padded = new_positions + self.fov_offsets_in_padded

        self.padded_env_matrix[old_positions_in_padded[:, 0],
                               old_positions_in_padded[:, 1], 1] = 0
        self.padded_env_matrix[new_positions_in_padded[:, 0],
                               new_positions_in_padded[:, 1], 1] = 1

    def update_obstacles_matrix(self, new_positions):
        """
        :param new_positions: 2d numpy array of shape (x, 2) where 0 <= x.
        :return: None.
        """
        old_positions = self.memory["env_vectors"]["found_obstacles"]

        self.env_matrix[old_positions[:, 0], old_positions[:, 1], 2] = 0
        self.env_matrix[new_positions[:, 0], new_positions[:, 1], 2] = 1

        old_positions_in_padded = old_positions + self.fov_offsets_in_padded
        new_positions_in_padded = new_positions + self.fov_offsets_in_padded

        self.padded_env_matrix[old_positions_in_padded[:, 0],
                               old_positions_in_padded[:, 1], 2] = 0
        self.padded_env_matrix[new_positions_in_padded[:, 0],
                               new_positions_in_padded[:, 1], 2] = 1

    @abstractmethod
    def get_action(self):
        # 1. Perceive.
        # -> env_vectors.
        self.env_vectors = \
            self.env.perceive_globally(self.idx_agent, is_prey=self.is_prey)
        self.own_position = self.env_vectors["own_position"].copy()

        # 2. Update matrix representation.
        # env_vectors -> env_matrix.
        self.update_env_matrix()

        # 3. Customized part.
        next_action = None

        # 4. Update memory.
        self.memory["env_vectors"] = copy.deepcopy(self.env_vectors)

        return next_action

    def get_next_position(self):
        next_action = None

        return next_action

    def get_open_axial_neighbors(self, position_concerned):
        """
        :param position_concerned: 1d numpy with the shape (2,).
        :return: 2d numpy array with the shape (x, 2), where 0 <= x <= 4,
                 depending on how many axial neighbors are still open,
                 i.e., not be occupied.
        """
        neighbors = self.axial_neighbors_mask + position_concerned

        open_idx = []
        for idx, neighbor in enumerate(neighbors):
            if not self.is_collide(neighbor):
                open_idx.append(idx)

        open_neighbors = neighbors[open_idx, :]

        return open_neighbors.copy()

    def is_collide(self, new_position):
        """
        Check the whether ``new_position`` collide with others in the global
        scope.

        ``new_position`` is valid
        if it additionally does not locate out the grid world boundaries.
        If it move out of the boundaries, it can also been seen that the agent
        collides with the boundaries, and so also a kind of collision.

        :param new_position: 1d numpy array with the shape (2,).
        :return: boolean, indicates  valid or not.
        """
        new_position = new_position + self.fov_offsets_in_padded
        pixel_values_in_new_position = \
            self.padded_env_matrix[new_position[0], new_position[1], :-1].sum()

        collide = False
        if pixel_values_in_new_position != 0:
            collide = True

        return collide

    def is_a_prey_captured(self, prey_position):
        """
        :param prey_position: 1d numpy array of shape (2,).
        :return: boolean
        """

        yes_no = True

        capture_positions = self.axial_neighbors_mask + prey_position + \
            self.fov_offsets_in_padded

        occupied_capture_positions = \
            self.padded_env_matrix[capture_positions[:, 0],
                                   capture_positions[:, 1], :-1].sum()

        if occupied_capture_positions != 4:
            yes_no = False

        return yes_no

    def random_walk(self):
        """
        :return: 1d numpy array of shape (2,).

        Prevent walk to the last position if the last position is not the
        only position it can walk to.
        """
        # 2d numpy array with the shape (x, 2), where 0 <= x <= 4.
        open_neighbors = self.get_open_axial_neighbors(self.own_position)

        if open_neighbors.shape[0] == 0:
            next_position = self.own_position
            return next_position

        # Otherwise.
        candidates = []
        last_position = self.memory["env_vectors"]["own_position"]
        for neighbor in open_neighbors:
            # Avoid returning to the position in the last time step.
            if neighbor.tolist() != last_position.tolist():
                candidates.append(neighbor)

        n_candidates = len(candidates)
        if n_candidates == 0:
            # Only one open neighbor, walk from it in the last time step.
            next_position = last_position.copy()
        else:
            # Random select one candidate.
            idx = np.random.choice(n_candidates, 1)[0]
            next_position = candidates[idx]

        return next_position


def test():
    pass


if __name__ == "__main__":
    test()
