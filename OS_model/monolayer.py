import numpy as np
from numpy import random
import math
from math import sqrt, log, exp
from matplotlib import pyplot as plt
from scipy import spatial
from .utils import random_unit_vector, set_random_cells, set_cell_sheet, generate_positions_array, \
    random_forces, generate_axes


class Monolayer:
    """Monolayer of cells in 2D space"""

    def __init__(self, p=0.5, size=10, rand=False, n=50, space=False):
        """
        Initialises the monolayer with cells of diameter 1, and a reflective boundary condition.

        Parameters
        ----------
        p : int, float, optional
            Proportion of cells which are of Type 0, taking values in [0,1]. Default is 0.5.

        size : int, optional
            Size of the monolayer (maximum x and y coordinate for the cell centres). Default is 10.

        rand : bool, optional
            Determines the initial configuration type in the monolayer. True corresponds to a random distribution
            of cells in the monolayer, whereas False corresponds to a hexagonal lattice structure. Default is False.

        n : int, optional
            Number of cells in the monolayer. Only used when rand = True. Default is 50.

        space: bool, optional
            Determines if additional spacing is desired around the monolayer. If False, a square domain (0, size)^2
            is created, with reflective boundaries. If True, this domain is expanded by 2 cell diameters.
            Default is False.
        """
        self.mu = 50
        self.lam = 0.1
        self.k_c = 5
        self.k_pert = 1
        self.sim_time = 0
        self.r_max = 2.5
        self.mag = 0.05
        self.drag = 1
        self.time_step = 0.005
        self.size = size
        if rand:
            self.num_cells = n
            self.initial_positions = set_random_cells(n, size)
        else:
            self.initial_positions = set_cell_sheet(radius=0.5, size=size)
        self.positions = generate_positions_array(self.initial_positions)
        if not rand:
            self.num_cells = len(self.positions)
        self.type_0 = round(self.num_cells * p)
        self.type_1 = self.num_cells - self.type_0
        self.cell_types = np.asarray([0] * self.type_0 + [1] * self.type_1)
        if not rand:
            random.shuffle(self.cell_types)
        self.cell_radius = np.asarray([0.5] * self.num_cells)
        self.initial_fractional_length = self.fractional_length()
        self.division_timer = None
        self.division_rates = None
        if space:
            self.space = 4
        else:
            self.space = 0

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
        Initialises scale factor by which to multiply the spring constant in the attractive heterotypic interaction.

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
        Initialises scale factor by which to multiply the random perturbation force acting on cells in the monolayer.

        Parameters
        ----------
        k_pert : int, float
            Multiplicative scaling factor of the perturbation magnitude. Default value is 1.
        """
        self.k_pert = k_pert

    def set_time_step(self, time_step):
        """
        Initialises the time step for the simulation.

        Parameters
        ----------
        time_step : float
            Simulation time step. Default is 0.005 hours.
        """
        self.time_step = time_step

    def set_r_max(self, r_max):
        """
        Initialises the maximum interaction distance for cells in the monolayer.

        Parameters
        ----------
        r_max : int, float
            The maximum euclidean distance permitting interaction between two cells. Default is 2.5 cell diameters.
        """
        self.r_max = r_max

    def set_mag(self, mag):
        """
        Initialises the base magnitude of perturbation for the simulation.

        Parameters
        ----------
        mag : int, float
            Base magnitude of perturbation. Default is 0.05.
        """
        self.mag = mag

    def set_drag(self, drag):
        """
        Initialises the drag/viscosity coefficient.

        Parameters
        ----------
        drag : int, float
            The drag coefficient. Default is 1.
        """
        self.drag = drag

    def set_space(self):
        """
        Adds empty space around the monolayer, expanding the simulation domain by 2 cell diameters.
        Allows for more freedom of movement during simulation.
        Note: Equivalent of specifying space = True on initialising the monolayer.
        """
        self.space = 4

    def manual_cell_placement(self, coordinates, types):
        """
        Helper function which allows manual cell placement for initial cell configuration to aid in testing.

        Parameters
        ----------
        coordinates: tuple
            A tuple of tuples, containing the desired starting cell coordinates.

        types: list, tuple
            A list or tuple containing the desired cell types.
        """
        self.initial_positions = coordinates
        self.positions = generate_positions_array(self.initial_positions)
        self.num_cells = len(self.positions)
        self.cell_types = np.asarray(types)
        self.type_1 = sum(self.cell_types)
        self.type_0 = self.num_cells - self.type_1
        self.cell_radius = np.asarray([0.5] * self.num_cells)

    def neighbours(self, radius):
        """
        Generates a list of the neighbours of each cell in the monolayer, where a neighbour is
        another cell in the monolayer whose centre is within the chosen interaction radius of the specified cell.

        Returns
        -------
        list
            A list of lists, with the ith list being a list of the neighbours of cell i.
        """
        cell_positions = self.positions
        cell_tree = spatial.KDTree(cell_positions)
        neighbours = cell_tree.query_ball_tree(cell_tree, r=radius)
        return neighbours

    def fractional_length(self):
        """
        Calculates the fractional length, a measure of sorting, for the monolayer in its current state.

        Returns
        -------
        fractional_length : float
            A fraction indication the proportion of edge contact of cells that is heterotypic (between two cells
            of different types).
        """
        neighbours = self.neighbours(2 * max(self.cell_radius))
        total_edge_contact, het_edge_contact = 0, 0
        for cell_a_index in range(self.num_cells):
            cell_a = self.positions[cell_a_index]
            a_type = self.cell_types[cell_a_index]
            radius_a = self.cell_radius[cell_a_index]
            for cell_b_index in neighbours[cell_a_index]:
                if cell_b_index > cell_a_index:
                    cell_b = self.positions[cell_b_index]
                    b_type = self.cell_types[cell_b_index]
                    radius_b = self.cell_radius[cell_b_index]
                    dist = np.linalg.norm(cell_a - cell_b)
                    if dist == 0:
                        dist = 0.0001
                    natural_separation = radius_a + radius_b
                    if natural_separation > dist:  # If we have overlapping cells, calculate edge length
                        s = 0.5 * (dist + radius_a + radius_b)
                        area = sqrt(s * (s - dist) * (s - radius_a) * (s - radius_b))
                        edge_length = 4 * area / dist
                        total_edge_contact += edge_length
                        if b_type != a_type:  # If cell_a and cell_b are not the same type
                            het_edge_contact += edge_length
        if total_edge_contact == 0:
            total_edge_contact = 0.0001
        fractional_length = het_edge_contact / total_edge_contact
        return fractional_length

    def interaction_forces(self):
        """
        Generates the interaction forces acting on all n cells in the monolayer, caused by the other cells
        present in the monolayer.

        Returns
        -------
        numpy.array
            An n x 2 array, where n is the number of cells in the monolayer. The ith entry contains
            the total 2D force acting on cell i caused by neighbouring cells.
        """
        neighbours = self.neighbours(self.r_max)
        forces = np.zeros((self.num_cells, 2))
        for cell_a_index in range(self.num_cells):
            cell_a = self.positions[cell_a_index]
            a_type = self.cell_types[cell_a_index]
            for cell_b_index in neighbours[cell_a_index]:
                if cell_b_index > cell_a_index:
                    cell_b = self.positions[cell_b_index]
                    b_type = self.cell_types[cell_b_index]
                    dist = np.linalg.norm(cell_a - cell_b)
                    if dist == 0:
                        dist += 0.0001
                    r = cell_b - cell_a  # Vector from cell a to cell b
                    r_hat = r / dist  # Unit vector
                    natural_separation = self.cell_radius[cell_a_index] + self.cell_radius[cell_b_index]
                    separation = dist - natural_separation
                    mu = self.mu
                    if separation < 0:  # If we have overlapping cells, regardless of type, repulsion occurs
                        f = mu * natural_separation * r_hat * log(1 + separation / natural_separation)
                    else:  # If cells are not overlapping but are within interaction radius
                        if b_type != a_type:  # If cell_a and cell_b are not the same type
                            mu *= self.lam  # Use heterotypic spring constant
                        f = mu * separation * r_hat * exp(-self.k_c * separation / natural_separation)
                    forces[cell_a_index] += f
                    forces[cell_b_index] -= f
        return forces

    def simulate_step(self):
        """
        Simulates one time step of the model, using a forward-Euler time step equation, and updates the
        position of each cell in the monolayer accordingly. Implements reflective BCs when there is no cell
        division, and free BCs otherwise.
        """
        if self.division_timer is not None:
            self.cell_division()
        radius = max(self.cell_radius)
        positions_for_update = np.zeros_like(self.positions)
        interaction_forces = self.interaction_forces()
        scale = sqrt(2 * self.k_pert * self.mag / self.time_step)
        rand_forces = random_forces(scale=scale, shape=self.positions.shape)
        for cell_index in range(self.num_cells):
            current_position = self.positions[cell_index]
            net_force = interaction_forces[cell_index] + rand_forces[cell_index]
            new_position = current_position + self.time_step * net_force / self.drag
            if self.division_timer is None:
                for i in range(2):  # Implementing no flux reflective boundary condition
                    if new_position[i] < -self.space * radius:
                        new_position[i] = -2 * self.space * radius - new_position[i]
                    elif new_position[i] > self.size + self.space * radius:
                        new_position[i] = 2 * (self.size + self.space * radius) - new_position[i]
            positions_for_update[cell_index] = new_position
        self.positions = positions_for_update
        self.sim_time += self.time_step

    def reset(self):
        """
        Resets the entire monolayer to the initial position state, setting the simulation time count back to 0.
        """
        self.positions = generate_positions_array(self.initial_positions)
        self.sim_time = 0
        self.num_cells = len(self.initial_positions)

    def simulate(self, end_time):
        """
        Simulates the model from its current state until 'end_time' (or from time 0 until 'end_time' if this time has
        already been passed), using a forward-Euler time step equation. Will not compute any
        additional outputs eg fractional length, or plot any visualisations.

        Parameters
        ----------
        end_time : int, float
            The end time of the simulation, in hours.

        """
        if end_time < self.sim_time:  # If the desired simulation time has already been passed, reset cells
            self.reset()
        if end_time != self.sim_time:
            length = end_time - self.sim_time  # Calculate remaining time needed to run simulation
            its = int(length / self.time_step)  # Calculate number of iterations needed for end time to be reached
            for _ in range(its):
                self.simulate_step()

    def show_cells(self, show_interactions=False, file_name=None):
        """
        Produces a visual representation of the current cell configuration of the monolayer.

        Parameters
        ----------
        show_interactions : bool
            'True' will show the interaction areas of each cell. Default is False.

        file_name : str
            String of desired file name, in the format 'file_name.pdf'. If None, will not save.
            Default is None.
        """
        cell_colour = ['plum', 'royalblue']
        radius = max(self.cell_radius)
        size = self.size
        spacing = self.space + 1
        if self.division_timer is not None and self.sim_time != 0:
            spacing += np.mean(self.division_rates) * self.sim_time * self.size  # Keep cells in plot window
        fig, ax = generate_axes(show_interactions=show_interactions, radius=radius, size=size,
                                spacing=spacing)
        cell_index = 0
        for xi, yi, cell_type in zip(self.positions[:, 0], self.positions[:, 1], self.cell_types):
            if show_interactions:
                interaction_zone = plt.Circle((xi, yi), radius=self.r_max, facecolor='grey', edgecolor='k', alpha=0.15)
                fig.gca().add_artist(interaction_zone)
            cell = plt.Circle((xi, yi), radius=self.cell_radius[cell_index],
                              facecolor=cell_colour[cell_type], edgecolor='k')
            fig.gca().add_artist(cell)
            cell_index += 1
        plt.title('Cells at ' + str(round(self.sim_time, 1)) + ' hours')
        plt.show()
        if file_name is not None:
            fig.savefig(file_name, bbox_inches='tight', )

    def measure_sorting(self, end_time):
        """
        Simulates the model from its current state until 'end_time' (or from time 0 until 'end_time' if this time has
        already been passed), recording the fractional length at each time step.

        Note, if the model has an initial fractional length of 0 (fully sorted) then fractional lengths
        returned are not normalised.

        Parameters
        ----------
        end_time : int, float
            The end time of the simulation, in hours.

        Returns
        -------
        fractional_length : np.ndarray
            An array containing the fractional lengths at each time step in the second row,
            normalised by the fractional length at time 0, and the corresponding time values in the first.
        """
        normalising_constant = self.initial_fractional_length
        if normalising_constant == 0:
            normalising_constant = 1
        if end_time < self.sim_time:  # If the desired simulation time has already been passed, reset cells
            self.reset()
        length = end_time - self.sim_time  # Calculate remaining time needed to run simulation for
        if length == 0:
            its = 0
        else:
            its = math.ceil(length / self.time_step)  # Calculate number of iterations needed for end time to be reached
        frac_length = np.zeros((2, its + 1))
        for i in range(its + 1):
            frac_length[0, i] = self.sim_time
            frac_length[1, i] = self.fractional_length() / normalising_constant
            if length != 0 and i != its:
                self.simulate_step()
        return frac_length

    def set_division_timer(self, division_rate, division_rate_1=None):
        """
        Initialises cell division process in the monolayer, setting an exponentially distributed division
        time for each cell.

        Parameters
        ----------
        division_rate : int, float
            The division rate, in divisions per hour. Default is 1/24, corresponding to a cell cycle length of 24 hours.
            Note: Should be strictly positive.

        division_rate_1 : int, float, None
            The division rate, in divisions per hour. If None, both cell types are taken to have the same division rate.
            Default is None. Note: Should be strictly positive.

        """
        if division_rate_1 is None:
            division_rate_1 = division_rate
        self.division_rates = self.cell_types * (division_rate_1 - division_rate) + division_rate
        cell_clocks = np.zeros(self.num_cells)
        for index, rate in enumerate(self.division_rates):
            cell_clocks[index] = random.exponential(1/rate)
        self.division_timer = cell_clocks

    def cell_division(self):
        """
        Checks the current division status of each cell in the monolayer, and implements cell division for any cells
        ready to divide. Daughter cells are clones of the parent cell, placed in a random direction 0.25 cell diameters
        away from the parent.
        """
        initial_cell_count = self.num_cells
        epsilon = 0.5 * max(self.cell_radius)
        for cell_index in range(initial_cell_count):
            division_time = self.division_timer[cell_index]
            if division_time <= 0:  # If cell is ready to divide
                self.division_timer[cell_index] = random.exponential(
                    1/self.division_rates[cell_index])  # Reset mother cell timer
                new_cell = self.positions[cell_index] + epsilon * random_unit_vector()
                self.positions = np.append(self.positions, [new_cell], axis=0)
                self.num_cells += 1
                self.cell_radius = np.append(self.cell_radius,
                                             0.5)  # Daughter cell has same size as other cells in monolayer
                self.cell_types = np.append(self.cell_types,
                                            self.cell_types[cell_index])  # Daughter cell same type as mother cell
                self.division_timer = np.append(self.division_timer, random.exponential(
                    1/self.division_rates[cell_index]))  # Add timer for daughter
                self.division_rates = np.append(self.division_rates,
                                                self.division_rates[cell_index])  # Add division rate for daughter cell
                self.type_1 = sum(self.cell_types)  # Update numbers of cell types
                self.type_0 = self.num_cells - self.type_1
            else:
                self.division_timer[cell_index] -= self.time_step  # Count down by time step

    def manual_division_timer(self, division_times, division_rates):
        """
        Helper function to aid in tests. Allows manual setting of first division times and division rates of cells.
        in the monolayer.

        Parameters
        ----------
        division_times : tuple, list
            A tuple or list of the desired first division times of the cells in the monolayer.

        division_rates : tuple, list
            A tuple or list of the desired division rates of the cells in the monolayer.
        """
        self.division_timer = np.asarray(division_times)
        self.division_rates = np.asarray(division_rates)
