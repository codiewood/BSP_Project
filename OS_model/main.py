import numpy as np
from numpy import random
import math
from math import sqrt, log, exp
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from scipy import spatial
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
    float
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
    tuple
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
        self.mu = 50
        self.lam = 0.1
        self.k_c = 5
        self.k_pert = 1
        self.r0 = 1
        self.r1 = 1
        self.sim_time = 0
        self.sim_params = (2.5, 0.05, 1)
        self.initial_positions = self.set_random_cells()
        self.positions = self.generate_initial_positions_array()
        self.cell_types = [0] * self.type_0 + [1] * self.type_1

    def set_mu(self, mu):
        """
        Initialises spring constant of the cells in the monolayer.

        Parameters
        ----------
        mu : int, float
            Spring constant. Default value is 50.
        """
        self.mu = mu

    def set_lam(self, lam):
        """
        Initialises scale factor by which to multiply the spring constant in the heterotypic interaction.

        Parameters
        ----------
        lam : int, float
            Multiplicative scaling factor used to determine heterotypic spring constant. Default value is 0.1.
        """
        self.lam = lam

    def set_k_c(self, k_c):
        """
        Initialises decay of attraction force of the cells in the monolayer.

        Parameters
        ----------
        k_c : int, float
            Decay of attraction force. Default value is 5.
        """
        self.k_c = k_c

    def set_k_pert(self, k_pert):
        """
        Initialises decay of attraction force of the cells in the monolayer.

        Parameters
        ----------
        k_pert : int, float
            Multiplicative scaling factor of the perturbation magnitude. Default value is 1.
        """
        self.k_pert = k_pert

    def set_radius(self, rad1):
        """
        Initialises differing radii of the cells in the monolayer. Radius of cell type 0 is taken as 1 unit.

        Parameters
        ----------
        rad1 : int, float
            Radius of cells with type 1, relative to that of cell type 0. Default value is 1.
        """
        self.r1 = rad1

    def simulation_parameters(self, r_max, mag, drag):
        """
        Initialises the simulation parameters.

        Parameters
        ----------
        r_max : int, float, optional
            The maximum euclidean distance permitting interaction between two cells. Default is 2.5.

        mag : int, float, optional
            Magnitude of perturbation. Default is 0.05.

        drag : int, float, optional
            The drag coefficient. Default is 1.
        """
        self.sim_params = (r_max, mag, drag)

    def set_random_cells(self):
        """
        Generates a tuple of coordinates for the initial positions of all cells within the monolayer,
        where each cell centre is random (uniformly distributed).

        Returns
        -------
        tuple
            Tuple of tuples, representing the initial (random) coordinates of each cell.
        """
        cell_positions = ()
        for _ in range(self.num_cells):
            x = (uniform_coords(self.size),)
            cell_positions = cell_positions + x
        return cell_positions

    # def set_hexagonal_grid_cells(self):
    #     """
    #     Generates a tuple of coordinates for the initial positions of all cells within the monolayer,
    #     where cells are placed on a hexagonal grid.
    #
    #     Returns
    #     -------
    #     tuple
    #         Tuple of tuples, representing the initial (random) coordinates of each cell.
    #     """
    #     size = self.size
    #     radius = max(self.r0, self.r1)
    #     cell_positions = ((size/2, size/2),)
    #     for i in range(self.num_cells):
    #         xi, yi = cell_positions[i] - (2*radius, 0)
    #         x = ((xi,yi),)
    #         cell_positions = cell_positions + x
    #     return cell_positions

    def generate_initial_positions_array(self):
        """
        Generates a n x 2 array of coordinates for the initial positions of all n cells within the monolayer.
        Note this generates an array, which is mutable.

        Returns
        -------
        np.ndarray
            n x 2 array of lists, representing the coordinates of each cell.
        """
        mutable_positions = []
        for xi, yi in self.initial_positions:
            coords = np.array([xi, yi])
            mutable_positions.append(coords)
        return np.stack(mutable_positions)

    def neighbours(self):
        """
        Generates a list of the neighbours of each cell in the monolayer, where a neighbour is
        another cell in the monolayer whose centre is within the interaction radius (r_max) of the specified cell.

        Returns
        -------
        list
            A list of lists, with the ith list being a list of the neighbours of cell i.
        """
        cell_positions = self.positions
        cell_tree = spatial.KDTree(cell_positions)
        r_max = self.sim_params[0]
        neighbours = cell_tree.query_ball_tree(cell_tree, r=r_max)
        return neighbours

    def interaction_forces(self):
        """
        Generates the interaction forces acting on all n cells in the monolayer, caused by the other cells
        present in the monolayer.

        Returns
        -------
        numpy.array
            A n x n x 2 array, where n is the number of cells in the monolayer. The [i,j] entry contains
            the 2D force acting on cell i caused by cell j.
        """
        cell_positions = self.positions
        cell_types = self.cell_types
        cell_count = self.num_cells
        neighbours = self.neighbours()
        forces = np.zeros((cell_count, cell_count, 2))
        cell_a_index = 0
        while cell_a_index < cell_count:
            # for cell_a_index in range(cell_count): appears to be less efficient?
            cell_a = cell_positions[cell_a_index]
            a_type = cell_types[cell_a_index]
            for cell_b_index in neighbours[cell_a_index]:
                if cell_b_index != cell_a_index:
                    cell_b = cell_positions[cell_b_index]
                    b_type = cell_types[cell_b_index]
                    dist = euclidean(cell_a, cell_b)
                    r = cell_b - cell_a  # Vector from cell a to cell b
                    r_hat = r / dist  # Unit vector
                    natural_separation = (a_type + b_type) * (self.r1 - self.r0) + 2 * self.r0
                    gap = dist - natural_separation
                    mu = self.mu
                    if gap < 0:  # If we have overlapping cells, regardless of type, repulsion occurs
                        f = mu * r_hat * log(1 + gap / natural_separation)
                    else:  # If cells are not overlapping but are within interaction radius
                        if b_type != a_type:  # If cell_a and cell_b are not the same type
                            mu *= self.lam  # Use heterotypic spring constant
                        f = mu * gap * r_hat * exp(-self.k_c * gap / natural_separation)
                    forces[cell_a_index, cell_b_index] = f
            cell_a_index += 1
        return forces

    def simulate_step(self, time_step=0.005):
        """
        Simulates one time step of the model, using a forward-Euler time step equation, and updates the
        position of each cell in the monolayer accordingly.

        Parameters
        ----------
        time_step : int, float
            The time step of the simulation, in hours. Default is 0.005.
        """
        mag, drag = self.sim_params[1:3]
        k_pert = self.k_pert
        cell_count = self.num_cells
        positions_for_update = self.positions
        forces = self.interaction_forces()
        cell_index = 0
        while cell_index < cell_count:
            # for cell_index in range(cell_count): ??
            current_position = positions_for_update[cell_index]
            cell_forces = forces[cell_index]
            interaction_forces = sum(cell_forces[:])
            new_position = np.zeros(2)
            for i in range(2):
                net_force = interaction_forces[i] + rand_pert(k_pert * mag, time_step)
                new_position[i] = current_position[i] + np.round(time_step * net_force / drag, 3)
            if 0 <= new_position[0] <= self.size and 0 <= new_position[1] <= self.size:  # Accepting any moves in domain
                positions_for_update[cell_index] = new_position  # Note here this algorithm assumes all cells move
                # simultaneously which seems a reasonable assumption for small time steps
            cell_index += 1
        self.sim_time += time_step
        self.sim_time = round(self.sim_time, 3)

    def reset(self):
        """
        Resets the entire monolayer to the initial position state, setting the simulation time count back to 0.
        """
        self.positions = self.generate_initial_positions_array()
        self.sim_time = 0

    def simulate(self, end_time, time_step=0.005):
        """
        Simulates the model until 'end_time', using a forward-Euler time step equation.

        Parameters
        ----------
        end_time : int, float
            The end time of the simulation, in hours.

        time_step : int, float
            The time step of the simulation, in hours. Default is 0.005.

        """
        if end_time < self.sim_time:  # If the desired simulation time has already been passed, reset cells
            self.reset()
        elif end_time != self.sim_time:
            length = end_time - self.sim_time  # Calculate remaining time needed to run simulation for
            its = math.ceil(length / time_step)  # Calculate number of iterations needed for end time to be reached
            for _ in range(its):
                self.simulate_step(time_step)

    def show_cells(self, time=0, time_step=0.005, show_interactions=False):
        """
        Shows a visual representation of the cell configuration at time 'time'.
        Note: For computational efficiency, if plotting multiple times it is best to do so in chronological order.

        Parameters
        ----------
        time : int, float
            The time at which the cells are shown, in hours. Default is 0.

        time_step : int, float
            The time step of the simulation, in hours. Default is 0.005.

        show_interactions : bool
            'True' will show the interaction areas of each cell. Default is False.
        """
        vmax = self.size
        radius = max(self.r0, self.r1)
        cell_types = self.cell_types
        r_max = self.sim_params[0]
        fig, ax = plt.subplots()
        ax.set_xlim(-radius, vmax + radius)
        ax.set_ylim(-radius, vmax + radius)
        ax.set_aspect(1)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plum_patch = mpatches.Patch(facecolor='plum', edgecolor='k', label='Type 0')
        blue_patch = mpatches.Patch(facecolor='royalblue', edgecolor='k', label='Type 1')
        leg = [plum_patch, blue_patch]
        if show_interactions:
            int_patch = mpatches.Patch(facecolor='grey', edgecolor='k', alpha=0.15, label='Interaction')
            leg.append(int_patch)
        plt.legend(handles=leg, bbox_to_anchor=((3 - len(leg)) / 6, -0.15, len(leg) / 3, .102), loc='upper left',
                   ncol=len(leg), mode="expand", borderaxespad=0.)
        self.simulate(time, time_step)
        cell_coordinates = self.positions
        for xi, yi, ti in zip(cell_coordinates[:, 0], cell_coordinates[:, 1], cell_types):
            if show_interactions:
                interaction_zone = plt.Circle((xi, yi), radius=r_max,
                                              facecolor='grey', edgecolor='k', alpha=0.15)
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
        plt.show()
