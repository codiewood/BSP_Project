import numpy as np
from numpy import random
import math
from math import sqrt, log, exp
# from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean


def rand_pert(mag=0.05, time_step=0.005):
    """
    A function that takes a magnitude and time step and generates a random perturbation force.

    Parameters
    ----------
    mag : int, float
        Magnitude of perturbation. Default is 0.05.

    time_step : int, float
        The time step of the simulation, in hours. Default is 0.005.

    Returns
    -------
    numpy.array
        The random perturbation force.
    """
    x = random.normal(0, 1)
    force = sqrt(2 * mag / time_step) * x
    return force


def uniform_coords(lim):
    """
    Generates uniformly distributed random coordinates in the 2D square, (0,lim)^2.

    Parameters
    ----------
    lim : int, float
        Upper bound of coordinate value in both the x and y directions.

    Returns
    -------
    list
        2D coordinates.
    """
    x = round(random.uniform(lim), 3)
    y = round(random.uniform(lim), 3)
    coords = np.array([x, y])
    return coords


class Monolayer:
    """Monolayer of cells in 2D space"""
    def __init__(self, n, p=0.5, size=20):
        """
        Initialises properties of the monolayer.

        Parameters
        ----------
        n : int
            Number of cells present in the monolayer

        p : int, float
            Proportion of cells which are of Type 0, taking values in [0,1]. Default is 0.5.

        size : int, float
            Size of the monolayer (maximum x and y coordinate for the cell centres). Default is 20.
        """
        self.num_cells = n
        self.size = size
        self.type_0 = round(n * p)
        self.type_1 = n - self.type_0
        self.initial_positions = self.generate_cells()

    def generate_cells(self):
        cell_positions = []
        for i in list(range(self.num_cells)):
            x = uniform_coords(self.size)
            cell_positions.append(x)
        return cell_positions
