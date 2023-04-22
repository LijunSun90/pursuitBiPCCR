"""
fitness_single_prey_encircle.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rewrite the codes in
https://github.com/LijunSun90/pursuitCCPSOR/blob/master/lib/fitness_encircle.m

Author: Lijun SUN.
Date: Thu Sep 10 2020.
"""
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from lib.fitness.compute_uniformity_symmetry import compute_uniformity_symmetry


def fitness_single_prey_encircle(position_prey=None,
                                 position_predators_swarm=None,
                                 use_swarm_positions=True,
                                 position_current=None,
                                 position_other_predators=None,
                                 use_neighbors=False,
                                 position_neighbors=None,
                                 position_neighbor_predators=None,
                                 position_neighbor_preys=None):
    """
    :param position_current: 1d numpy array of shape (2,).
        the current position of the predator that is being evaluated.
    :param position_other_predators: 2d numpy array of shape
        (n_predators - 1, 2).
    :param position_prey: 1d numpy array of shape (2,).
    :param use_swarm_positions: boolean.
        If True, this function evaluate the whole predators swarm,
        instead of a single predator.
    :param position_predators_swarm: 2d numpy array of shape (n_predators, 2).
    :param use_neighbors: boolean.
        If True, consider the neighbors including the inner-cluster other
        predators and preys and the other agents in other clusters.
    :param position_neighbors: 2d numpy array of shape (n_neighbors, 2).
    :param position_neighbor_predators:
        2d numpy array of shape (n_neighbor_predators, 2).
    :param position_neighbor_preys:
        2d numpy array of shape (n_neighbor_preys, 2).
    :return: a float.
    """

    if use_swarm_positions:
        position_predators = position_predators_swarm
    else:
        position_predators = \
            np.vstack((position_other_predators, position_current))

    # Define variables.
    # v1: 2. v2: 1.
    min_distance_with_predator = 2
    min_distance_with_prey = 1

    # ##################################################
    # Part - collision avoidance.

    # For the evaluation of the predators swarm,
    # the collision avoidance fitness is not calculated,
    # since it is designed to evaluate a single predator.
    if use_swarm_positions:
        fitness_repel = 1
    else:
        # Reshape the 1d array to 2d array.
        position_current = position_current.reshape(1, -1)

        if not use_neighbors:
            position_neighbor_predators = position_other_predators
            position_neighbor_preys = position_prey.reshape(1, -1)

        distance_with_other_predators = \
            pairwise_distances(position_current,
                               position_neighbor_predators,
                               metric='manhattan')

        distance_with_prey = \
            pairwise_distances(position_current,
                               position_neighbor_preys,
                               metric='manhattan')

        NND_with_other_predators = np.min(distance_with_other_predators)
        NND_with_preys = np.min(distance_with_prey)

        if min_distance_with_predator == min_distance_with_prey:
            min_distance = min_distance_with_predator
            NND = min(NND_with_other_predators, NND_with_preys)
            if NND <= min_distance:
                # 2.
                fitness_repel = np.exp(-5 * (NND - min_distance))
            else:
                fitness_repel = 1

        else:

            # NND is not due to a near predator.
            if NND_with_other_predators <= min_distance_with_predator:
                # 2.
                fitness_repel = np.exp(-5 * (NND_with_other_predators -
                                             min_distance_with_predator))
            else:
                fitness_repel = 1

            # NND is not due to a near prey.
            if NND_with_preys <= min_distance_with_prey:
                # 2.
                fitness_repel *= np.exp(-10 * (NND_with_preys -
                                               min_distance_with_prey))
            else:
                fitness_repel *= 1

    # ##################################################
    # Part - closure: the group center is located inside the convex hull.

    # 1d numpy array.
    try:
        hull = ConvexHull(position_predators)
        # 2d numpy array.
        hull_vertices = position_predators[hull.vertices, :]
    except QhullError:
        # When all points are in line.
        hull_vertices = position_predators.copy()

    polygon = Polygon(hull_vertices)
    point_prey = Point(position_prey)
    is_in = polygon.contains(point_prey)

    fitness_closure = 1 - is_in

    # ##################################################
    # Part - group expanse.

    fitness_expanse = \
        np.mean(
            np.linalg.norm(position_predators - position_prey, ord=2, axis=1))

    # ##################################################
    # Part - uniformity.

    fitness_uniformity = compute_uniformity_symmetry(position_predators,
                                                     position_prey)
    # Output.
    fitness = fitness_repel * (
            fitness_closure + fitness_expanse + fitness_uniformity)

    return fitness


def test():
    pass


if __name__ == "__main__":
    test()
