import numpy as np
from numpy import random
from math import sqrt, floor
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches


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


def random_unit_vector():
    """
    Generates a random 2D unit vector, from a standard bivariate normal distribution.

    Returns
    -------
    numpy.array
        2D vector.
    """
    x, y = random.normal(size=2)
    mag = sqrt(x ** 2 + y ** 2)
    return np.asarray([x / mag, y / mag])


def generate_cell_rows(radius, size, n):
    """
        Generates a tuple of coordinates for the initial positions of two rows of cells within the monolayer,
        where cells are packed into a hexagonal lattice.

        Parameters
        ----------
        radius : int, float
            The size of the cells being placed.

        size : int
            The desired length of the first row of cells, in number of cells.

        n : int
            The iteration index of the function, as it is called by set_cell_sheet. Determines the coordinates of the
            first cell placed. n = 0 will start from the lower left corner of the domain.

        Returns
        -------
        tuple
            Tuple of tuples, representing the initial coordinates of each cell in a set of two cell rows.
        """
    x = radius
    y = radius * (sqrt(3) * 2 * n + 1)
    row_switches = 0
    cell_rows = ()
    while radius <= x <= size - radius:
        if radius <= y <= size - radius:
            new_cell = ((x, y),)
            cell_rows = cell_rows + new_cell
        x += radius
        if row_switches / 2 == row_switches // 2:
            y += sqrt(3) * radius
        else:
            y -= sqrt(3) * radius
        row_switches += 1
    return cell_rows


def set_cell_sheet(radius, size):
    """
        Generates a tuple of coordinates for the initial positions of all cells within the monolayer,
        where cells are packed into a hexagonal lattice.

        Parameters
        ----------
        radius : int, float
            The size of the cells being placed.

        size : int
            The desired length of the first row of cells, in number of cells.

        Returns
        -------
        tuple
            Tuple of tuples, representing the initial coordinates of each cell in the monolayer.
        """
    cell_sheet = ()
    for row_set in range(floor(size / (2 * radius))):
        cell_sheet = cell_sheet + generate_cell_rows(radius, size, n=row_set)
    return cell_sheet


def set_random_cells(num_cells, size):
    """
        Generates a tuple of coordinates for the initial positions of all cells within the monolayer,
        where each cell centre is random (uniformly distributed).

        Returns
        -------
        tuple
            Tuple of tuples, representing the initial (random) coordinates of each cell.
        """
    cell_positions = ()
    for _ in range(num_cells):
        x = (uniform_coords(size),)
        cell_positions = cell_positions + x
    return cell_positions


def generate_initial_positions_array(initial_positions):
    """
        Generates a n x 2 array of coordinates from a tuple of n 2D tuples.
        Note this generates an array, which is mutable.

        Returns
        -------
        np.ndarray
            n x 2 array, representing the coordinates of each cell.
        """
    mutable_positions = []
    for xi, yi in initial_positions:
        coords = np.array([xi, yi])
        mutable_positions.append(coords)
    return np.stack(mutable_positions)


def random_forces(scale, shape):
    """
    A function that generates random perturbation forces.

    Returns
    -------
    np.array
        An n x 2 array of the random perturbation forces acting on each cell in the monolayer.
    """
    force = random.normal(0, 1, shape)
    return scale * force


def generate_axes(radius, size, spacing, show_interactions=False):
    """
        Generates the axes and plot onto which cells can be drawn by the show_cells function.

        Parameters
        ----------
        radius: int, float

        size : int

        spacing: int

        show_interactions : bool
            'True' will show the interaction areas (with radius r_max) of each cell. Default is False.

        Returns
        -------
        fig, ax :
            The figure and axis required for plotting.
        """
    fig, ax = plt.subplots()
    ax.set_xlim(-spacing * radius, size + spacing * radius)
    ax.set_ylim(-spacing * radius, size + spacing * radius)
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
