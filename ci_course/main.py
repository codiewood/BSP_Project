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
        self.positions = self.initial_positions
        self.cell_types = [0] * self.type_0 + [1] * self.type_1

    def generate_cells(self):
        """
        Generates a list of initial positions for all cells within the monolayer.

        Returns
        -------
        list
            List of tuples, representing the initial (random) coordinates of each cell.
        """
        cell_positions = []
        for i in list(range(self.num_cells)):
            x = uniform_coords(self.size)
            cell_positions.append(x)
        return cell_positions

    def neighbours(self, cell_index, r_max=2.5):  # May be redundant
        """
        Generates a list of the neighbours of a specified cell, where a neighbour is
        another cell in the monolayer whose centre is within the interaction radius (r_max) of the specified.

        Parameters
        ----------
        cell_index : int
            The position index of the cell within the monolayer whose neighbours we seek.

        r_max : int, float
            The maximum euclidean distance permitting interaction between two cells. Default is 2.5.

        Returns
        -------
        list
            A list of the index values of each neighbour of cell_a.
        """
        neighbours = []
        index = -1
        cell_a = self.positions[cell_index]
        for cell_b in self.positions:
            index += 1
            if np.array_equal(cell_a, cell_b):
                continue
            dist = euclidean(cell_a, cell_b)
            if dist < r_max:
                neighbours.append(index)
        return neighbours

    def interaction_forces(self, cell_index, mu=50, k_c=5, r_max=2.5):
        cell_a = self.positions[cell_index]
        neighbours = self.neighbours(cell_index, r_max)
        forces = [0] * self.num_cells
        for index in neighbours:
            cell_b = self.positions[index]
            dist = euclidean(cell_a, cell_b)
            r = cell_b - cell_a
            r_hat = r/dist
            if dist < 2:
                f = mu * r_hat * log(1 + (dist - 2)/2)
            else:
                f = mu * (dist - 2) * r_hat * exp(-k_c * (dist - 2)/2)
            forces[index] = f
        return forces

    def simulate_step(self, time_step=0.005, mag=0.05, eta=1, mu=50, k_c=5, r_max=2.5):
        index = -1
        updated_positions = self.positions
        for position in self.positions:
            index += 1
            int_forces = sum(self.interaction_forces(index, mu, k_c, r_max))
            net_force = int_forces + rand_pert(mag, time_step)
            new_position = position + time_step * net_force / eta
            if 0 <= new_position[0] <= self.size and 0 <= new_position[0] <= self.size:  # Accepting any moves in domain
                updated_positions[index] = new_position
        self.positions = updated_positions

    def simulate(self, end_time=100, time_step=0.005, mag=0.05, eta=1, mu=50, k_c=5, r_max=2.5):
        its = math.ceil(end_time/time_step)
        for i in list(range(its)):
            self.simulate_step(time_step, mag, eta, mu, k_c, r_max)
