import numpy as np
from numpy import random
import math
from math import sqrt, log, exp, floor
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from scipy import spatial
from scipy.spatial.distance import euclidean


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

    def __init__(self, p=0.5, size=10, rand=False, n=50):
        """
        Initialises the monolayer with cells of diameter 1.

        Parameters
        ----------
        p : int, float
            Proportion of cells which are of Type 0, taking values in [0,1]. Default is 0.5.

        size : int
            Size of the monolayer (maximum x and y coordinate for the cell centres). Default is 10.

        rand : bool
            Determines the initial configuration type in the monolayer. True corresponds to a random distribution
            of cells in the monolayer, whereas False corresponds to a hexagonal lattice structure. Default is False.

        n : int
            Number of cells in the monolayer. Only used when rand = True. Default is 50.
        """
        self.mu = 50
        self.lam = 0.1
        self.k_c = 5
        self.k_pert = 1
        self.sim_time = 0
        self.sim_params = (2.5, 0.05, 1)
        self.size = size
        if rand:
            self.num_cells = n
            self.initial_positions = self.set_random_cells()
        else:
            self.initial_positions = self.set_cell_sheet()
        self.positions = self.generate_initial_positions_array()
        if not rand:
            self.num_cells = len(self.positions)
        self.type_0 = round(self.num_cells * p)
        self.type_1 = self.num_cells - self.type_0
        self.cell_types = [0] * self.type_0 + [1] * self.type_1
        if not rand:
            random.shuffle(self.cell_types)
        self.cell_radius = [0.5] * self.num_cells

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
        Initialises scale factor by which to multiply the random perturbation force acting on cells in the monolayer.

        Parameters
        ----------
        k_pert : int, float
            Multiplicative scaling factor of the perturbation magnitude. Default value is 1.
        """
        self.k_pert = k_pert

    def simulation_parameters(self, r_max, mag, drag):
        """
        Initialises the simulation parameters.

        Parameters
        ----------
        r_max : int, float
            The maximum euclidean distance permitting interaction between two cells. Default is 2.5 cell diameters.

        mag : int, float
            Base magnitude of perturbation. Default is 0.05.

        drag : int, float
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

    def generate_cell_rows(self, n):
        """
        Generates a tuple of coordinates for the initial positions of two rows of cells within the monolayer,
        where cells are packed into a hexagonal lattice.

        Parameters
        ----------
        n : int
            The iteration index of the function, as it is called by set_cell_sheet. Determines the coordinates of the
            first cell placed. n = 0 will start from the lower left corner of the domain.

        Returns
        -------
        tuple
            Tuple of tuples, representing the initial coordinates of each cell in a set of two cell rows.
        """
        radius = 0.5
        x = radius
        y = radius * (sqrt(3) * 2 * n + 1)
        y = round(y, 3)
        row_switches = 0
        cell_rows = ()
        while radius <= x <= self.size - radius:
            if radius <= y <= self.size - radius:
                new_cell = ((x, y),)
                cell_rows = cell_rows + new_cell
            x += radius
            if row_switches / 2 == row_switches // 2:
                y += sqrt(3) * radius
            else:
                y -= sqrt(3) * radius
            y = round(y, 3)
            row_switches += 1
        return cell_rows

    def set_cell_sheet(self):
        """
        Generates a tuple of coordinates for the initial positions of all cells within the monolayer,
        where cells are packed into a hexagonal lattice.

        Returns
        -------
        tuple
            Tuple of tuples, representing the initial coordinates of each cell in the monolayer.
        """
        radius = 0.5
        cell_sheet = ()
        for row_set in range(floor(self.size / (2 * radius))):
            cell_sheet = cell_sheet + self.generate_cell_rows(row_set)
        return cell_sheet

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

    def neighbours(self, radius):
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
        neighbours = cell_tree.query_ball_tree(cell_tree, r=radius)
        return neighbours

    def cell_sorting_proportions(self):
        props = np.zeros(self.num_cells)
        neighbours = self.neighbours(2*max(self.cell_radius))
        for cell_index in range(self.num_cells):
            surrounding_cells = neighbours[cell_index]
            cell_type = self.cell_types[cell_index]
            surrounding_types = []
            for cell in surrounding_cells:
                surrounding_types.append(self.cell_types[cell])
            type_1_count = sum(surrounding_types[:])
            total_cell_count = len(surrounding_cells)
            if cell_type == 0:
                prop = (total_cell_count - type_1_count)/total_cell_count
            else:
                prop = type_1_count/total_cell_count
            props[cell_index] = prop
        return props

    def average_sorting_metric(self):
        props = self.cell_sorting_proportions()
        average_proportion = sum(props) / len(props)
        return average_proportion

    def cutoff_sorting_metric(self):
        i = 0
        props = self.cell_sorting_proportions()
        for j in range(len(props)):
            if props[j] > 0.5:
                i += 1
        prop_over_cutoff = i / len(props)
        return prop_over_cutoff

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
        neighbours = self.neighbours(self.sim_params[0])
        forces = np.zeros((self.num_cells, self.num_cells, 2))
        for cell_a_index in range(self.num_cells):
            cell_a = self.positions[cell_a_index]
            a_type = self.cell_types[cell_a_index]
            for cell_b_index in neighbours[cell_a_index]:
                if cell_b_index > cell_a_index:
                    cell_b = self.positions[cell_b_index]
                    b_type = self.cell_types[cell_b_index]
                    dist = euclidean(cell_a, cell_b)
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
                    forces[cell_a_index, cell_b_index] = np.round(f, 5)
                    forces[cell_b_index, cell_a_index] = -np.round(f, 5)
        return forces

    def fractional_length(self):
        """
        Calculates the fractional length, a measure of sorting, for the monolayer.

        Returns
        -------
        fractional_length : float
            A fraction indication the proportion of edge contact of cells that is heterotypic (between two cells
            of different types).
        """
        neighbours = self.neighbours(2*max(self.cell_radius))
        total_edge_contact = 0
        het_edge_contact = 0
        for cell_a_index in range(self.num_cells):
            cell_a = self.positions[cell_a_index]
            a_type = self.cell_types[cell_a_index]
            radius_a = self.cell_radius[cell_a_index]
            for cell_b_index in neighbours[cell_a_index]:
                if cell_b_index > cell_a_index:
                    cell_b = self.positions[cell_b_index]
                    b_type = self.cell_types[cell_b_index]
                    radius_b = self.cell_radius[cell_b_index]
                    dist = euclidean(cell_a, cell_b)
                    if dist == 0:
                        dist += 0.0001
                    natural_separation = radius_a + radius_b
                    if natural_separation > dist:  # If we have overlapping cells, calculate edge length
                        s = 0.5*(dist + radius_a + radius_b)
                        area = sqrt(s*(s-dist)*(s-radius_a)*(s-radius_b))
                        edge_length = 4 * area / dist
                        total_edge_contact += edge_length
                        if b_type != a_type:  # If cell_a and cell_b are not the same type
                            het_edge_contact += edge_length
        fractional_length = het_edge_contact / total_edge_contact
        return fractional_length

    def random_forces(self, time_step):
        """
        A function that takes a time step and generates random perturbation forces for each cell in the monolayer.

        Parameters
        ----------
        time_step : int, float
            The time step of the simulation, in hours.

        Returns
        -------
        force : np.array
            An n x 2 array of the random perturbation forces acting on each cell in the monolayer,
            returned to 3 decimal places.
        """
        mag = self.sim_params[1]
        force = np.zeros_like(self.positions)
        for index in range(self.num_cells):
            coords = random.normal(0, 1, 2)
            force[index] = sqrt(2 * self.k_pert * mag / time_step) * coords
            force = np.round(force, 5)
        return force

    def simulate_step(self, time_step=0.005):
        """
        Simulates one time step of the model, using a forward-Euler time step equation, and updates the
        position of each cell in the monolayer accordingly.

        Parameters
        ----------
        time_step : int, float
            The time step of the simulation, in hours. Default is 0.005.
        """
        drag = self.sim_params[2]
        positions_for_update = np.zeros_like(self.positions)
        interaction_forces = self.interaction_forces()
        random_forces = self.random_forces(time_step)
        for cell_index in range(self.num_cells):
            current_position = self.positions[cell_index]
            cell_interaction_force = sum(interaction_forces[cell_index][:])
            net_force = cell_interaction_force + random_forces[cell_index]
            new_position = current_position + time_step * net_force / drag
            for i in range(2):  # Implementing no flux reflective boundary condition
                if new_position[i] < 0:
                    new_position[i] = -new_position[i]
                elif new_position[i] > self.size:
                    new_position[i] = 2 * self.size - new_position[i]
            positions_for_update[cell_index] = new_position
        self.positions = positions_for_update
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

    def generate_axes(self, show_interactions=False):
        """
        Generates the axes and plot onto which cells can be drawn by the show_cells function.

        Parameters
        ----------
        show_interactions : bool
            'True' will show the interaction areas (with radius r_max) of each cell. Default is False.

        Returns
        -------
        fig, ax :
            The figure and axis required for plotting.
        """
        vmax = self.size
        radius = max(self.cell_radius)
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
        return fig, ax

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
        r_max = self.sim_params[0]
        fig, ax = self.generate_axes(show_interactions)
        self.simulate(time, time_step)
        cell_index = 0
        for xi, yi, ti in zip(self.positions[:, 0], self.positions[:, 1], self.cell_types):
            if show_interactions:
                interaction_zone = plt.Circle((xi, yi), radius=r_max, facecolor='grey', edgecolor='k', alpha=0.15)
                fig.gca().add_artist(interaction_zone)
            if ti == 0:
                cell_colour = 'plum'
            else:
                cell_colour = 'royalblue'
            cell = plt.Circle((xi, yi), radius=self.cell_radius[cell_index], facecolor=cell_colour, edgecolor='k')
            fig.gca().add_artist(cell)
            cell_index += 1
        plt.title('Cells at ' + str(self.sim_time) + ' hours')
        plt.show()
