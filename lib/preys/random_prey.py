"""
random_prey.py
~~~~~~~~~~~~~~

Author: Lijun SUN.
Date: Thu Sep 10 2020.
~~~~~~~~~~~~~~~~~~~~~~
Modified: Tue Nov 17, 2020.
1. A little more concise of the code.
"""
import copy
import numpy as np

from lib.agents.matrix_agent_global_perception import \
    MatrixAgentGlobalPerception
from lib.environment.matrix_world import MatrixWorld


class RandomPrey(MatrixAgentGlobalPerception):
    def __init__(self, env, idx_prey, under_debug=False):
        super(RandomPrey, self).__init__(env, idx_prey, under_debug)

    def set_is_prey_or_not(self):
        self.is_prey = True

    def get_action(self):
        # 1. Perceive.
        # -> env_vectors.
        self.env_vectors = \
            self.env.perceive_globally(self.idx_agent, is_prey=True)
        self.own_position = self.env_vectors["own_position"].copy()

        # 2. Update matrix representation.
        # env_vectors -> env_matrix.
        self.update_env_matrix()

        # 3. Customized part.
        next_position = self.get_next_position()

        # Get the action.
        # 1d numpy array with the shape(2,), np.array([delta_x, delta_y]).
        direction = self.env.get_offsets(self.own_position, next_position)
        next_action = self.env.direction_action[tuple(direction)]

        # 4. Update memory.
        self.memory["env_vectors"] = copy.deepcopy(self.env_vectors)

        return next_action

    def get_next_position(self):

        open_neighbors = self.get_open_axial_neighbors(self.own_position)

        # Random select one.
        n_open_neighbors = open_neighbors.shape[0]
        if n_open_neighbors == 0:
            next_position = self.own_position.copy()
        else:
            idx = np.random.choice(n_open_neighbors, 1)[0]
            next_position = open_neighbors[idx, :]

        return next_position


def test():
    world_rows = 40
    world_columns = 40

    n_prey = 4
    n_predators = 4 * (n_prey + 1)

    env = MatrixWorld(world_rows, world_columns,
                      n_preys=n_prey, n_predators=n_predators)
    env.reset(set_seed=True, seed=0)

    # 0, 16, 5, 2
    idx_prey = 0
    prey = RandomPrey(env, idx_prey)

    env.render(is_save=True)


if __name__ == "__main__":
    test()
