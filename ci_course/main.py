import numpy as np
from numpy import random
from math import sqrt


def rand_pert(n, mag=0.05, time_step=0.005):
    """
    A function that takes a number of cells, magnitude and time step and generates a random perturbation force.

    Parameters
    ----------
    n : int
        Number of cells in simulation.

    mag : int, float
        Magnitude of perturbation. Default is 0.05.

    time_step : int, float
        The time step of the simulation, in hours. Default is 0.005.

    Returns
    -------
    numpy.ndarray
        The random perturbation force, an n x 1 array.
    """
    x = random.normal(0, 1, n)
    force = sqrt(2*mag/time_step)*x
    return force
