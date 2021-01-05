import numpy as np
from numpy import random
import math
from math import sqrt, log, exp
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from scipy.spatial.distance import euclidean


def rand_pert(mag, time_step):
    """
    A function that takes a magnitude and time step and generates a random perturbation force.

    Parameters
    ----------
    mag : int, float
        Magnitude of perturbation.

    time_step : int, float
        The time step of the simulation, in hours.

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
    numpy.array
        2D coordinates.
    """
    x = round(random.uniform(lim), 3)
    y = round(random.uniform(lim), 3)
    coords = (x, y)
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
        self.initial_positions = self.set_cells()
        self.positions = self.generate_positions()
        self.cell_types = [0] * self.type_0 + [1] * self.type_1
        self.mu = 50
        self.mu_het = 5
        self.k_c = 5
        self.r0 = 1
        self.r1 = 1
        self.sim_time = 0

    def set_mu(self, mu=50, mu_het=5):
        """
        Initialises spring constants of the cells in the monolayer.

        Parameters
        ----------
        mu : int, float
            Spring constant. Default value is 50.

        mu_het : int, float
            Heterotypic spring constant. Default value is 5.
        """
        self.mu = mu
        self.mu_het = mu_het

    def set_k_c(self, k_c=5):
        """
        Initialises decay of attraction force of the cells in the monolayer.

        Parameters
        ----------
        k_c : int, float
            Decay of attraction force. Default value is 5.
        """
        self.k_c = k_c

    def set_radius(self, rad0=1, rad1=1):
        """
        Initialises radius of the cells in the monolayer.

        Parameters
        ----------
        rad0 : int, float
            Radius of cells with type 0. Default value is 1.

        rad1 : int, float
            Radius of cells with type 0. Default value is 1.
        """
        self.r0 = rad0
        self.r1 = rad1

    def set_cells(self):
        """
        Generates a tuple of coordinates for the initial positions of all cells within the monolayer.

        Returns
        -------
        tuple
            Tuple of tuples, representing the initial (random) coordinates of each cell.
        """
        cell_positions = ()
        for _ in list(range(self.num_cells)):
            x = (uniform_coords(self.size),)
            cell_positions = cell_positions + x
        return cell_positions

    def generate_positions(self):
        # TODO: docstring
        pos = []
        for xi, yi in self.initial_positions:
            coords = np.array([xi, yi])
            pos.append(coords)
        return pos

    def neighbours(self, cell_index, r_max=2.5):  # May be redundant
        """
        Generates a list of the neighbours of a specified cell, where a neighbour is
        another cell in the monolayer whose centre is within the interaction radius (r_max) of the specified cell.

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

    def interaction_forces(self, cell_index, r_max=2.5):
        """
        Generates the interaction forces acting on a particular cell in the monolayer, caused by the other cells
        present in the monolayer.

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
        cell_a = self.positions[cell_index]
        neighbours = self.neighbours(cell_index, r_max)
        forces = [0] * self.num_cells
        for index_b in neighbours:
            cell_b = self.positions[index_b]
            dist = euclidean(cell_a, cell_b)
            r = cell_b - cell_a
            r_hat = r/dist  # Unit vector between cell centres
            a_type = self.cell_types[cell_index]
            if self.cell_types[index_b] != a_type:  # If cell_a and cell_b are not the same type
                mu = self.mu_het  # Use heterotypic spring constant
                s = self.r0 + self.r1  # Natural seperation of cells
            else:
                mu = self.mu  # Use spring constant
                s = 2 * (a_type * (self.r1 - self.r0) + self.r0)  # Natural seperation of cells
            if dist < s:
                f = mu * r_hat * log(1 + (dist - s)/s)
            else:
                f = mu * (dist - s) * r_hat * exp(-self.k_c * (dist - s)/s)
            forces[index_b] = f
        return forces

    def simulate_step(self, time_step=0.005, mag=0.05, drag=1, r_max=2.5):
        """
        Simulates one time step of the model, using a forward-Euler time step equation, and updates the
        position of each cell in the monolayer accordingly.

        Parameters
        ----------
        time_step : int, float
            The time step of the simulation, in hours. Default is 0.005.

        mag : int, float
            Magnitude of perturbation. Default is 0.05.

        drag : int, float
            The maximum euclidean distance permitting interaction between two cells. Default is 1.

        r_max : int, float
            The maximum euclidean distance permitting interaction between two cells. Default is 2.5.
        """
        index = -1
        updated_positions = self.positions
        for position in self.positions:
            index += 1
            int_forces = sum(self.interaction_forces(index, r_max))
            net_force = int_forces + rand_pert(mag, time_step)
            new_position = position + np.round(time_step * net_force / drag, 3)
            if 0 <= new_position[0] <= self.size and 0 <= new_position[1] <= self.size:  # Accepting any moves in domain
                updated_positions[index] = new_position
        self.positions = updated_positions
        self.sim_time += time_step
        self.sim_time = round(self.sim_time, 3)

    def reset(self):
        # TODO: docstring
        self.positions = self.generate_positions()
        self.sim_time = 0

    def simulate(self, end_time=100, time_step=0.005, mag=0.05, drag=1, r_max=2.5):
        """
        Simulates the model until 'end_time', using a forward-Euler time step equation.

        Parameters
        ----------
        end_time : int, float
            The end time of the simulation, in hours. Default is 100.

        time_step : int, float
            The time step of the simulation, in hours. Default is 0.005.

        mag : int, float
            Magnitude of perturbation. Default is 0.05.

        drag : int, float
            The maximum euclidean distance permitting interaction between two cells. Default is 1.

        r_max : int, float
            The maximum euclidean distance permitting interaction between two cells. Default is 2.5.
        """
        if end_time < self.sim_time:  # If the desired simulation time has already been passed, reset cells
            self.reset()
        length = end_time - self.sim_time  # Calculate remaining time needed to run simulation for
        its = math.ceil(length/time_step)  # Calculate number of iterations needed for end time to be reached
        for _ in list(range(its)):
            self.simulate_step(time_step, mag, drag, r_max)

    def show_cells(self, time=0, time_step=0.005, mag=0.005, drag=1, r_max=2.5, show_interactions=False):
        """
        Shows a visual representation of the cell configuration at time 'time'.
        Note: For computational efficiency, if plotting multiple times it is best to do so in chronological order.

        Parameters
        ----------
        time : int, float
            The time at which the cells are shown, in hours. Default is 0.

        time_step : int, float
            The time step of the simulation, in hours. Default is 0.005.

        mag : int, float
            Magnitude of perturbation. Default is 0.05.

        drag : int, float
            The maximum euclidean distance permitting interaction between two cells. Default is 1.

        r_max : int, float
            The maximum euclidean distance permitting interaction between two cells. Default is 2.5.

        show_interactions : bool
            'True' will show the interaction areas of each cell. Default is False.
        """
        # TODO: refactor
        vmax = self.size
        radius = max(self.r0, self.r1)
        cell_types = self.cell_types
        fig, ax = plt.subplots()
        ax.set_xlim(-radius, vmax + radius)
        ax.set_ylim(-radius, vmax + radius)
        ax.set_aspect(1)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        self.simulate(time, time_step, mag, drag, r_max)
        pos = np.stack(self.positions)
        for xi, yi, ti in zip(pos[:, 0], pos[:, 1], cell_types):
            if show_interactions:
                interaction_zone = plt.Circle((xi, yi), radius=r_max, facecolor='grey', edgecolor='k', alpha=0.15)
                fig.gca().add_artist(interaction_zone)
            if ti == 0:
                cell_colour = 'plum'
                cell_radius = self.r0
            else:
                cell_colour = 'royalblue'
                cell_radius = self.r1
            cell = plt.Circle((xi, yi), radius=cell_radius, facecolor=cell_colour, edgecolor='k')
            fig.gca().add_artist(cell)
        plt.title('Cells at ' + str(self.sim_time) + ' hours')
        plum_patch = mpatches.Patch(facecolor='plum', edgecolor='k', label='Type 0')
        blue_patch = mpatches.Patch(facecolor='royalblue', edgecolor='k', label='Type 1')
        leg = [plum_patch, blue_patch]
        if show_interactions:
            int_patch = mpatches.Patch(facecolor='grey', edgecolor='k', alpha=0.15, label='Interaction')
            leg.append(int_patch)
        plt.legend(handles=leg, bbox_to_anchor=((3-len(leg))/6, -0.15, len(leg)/3, .102), loc='upper left',
           ncol=len(leg), mode="expand", borderaxespad=0.)
        plt.show()


# TODO: Find way to get all of cell data in one big array, positions to be array rather than list etc.
# TODO: Stance on overlap?
