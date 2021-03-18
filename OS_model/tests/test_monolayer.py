import OS_model
import numpy as np
from numpy import random
from math import sqrt


def test_set_mu():
    """
    Test the function `Monolayer.set_mu` in monolayer.py
    """
    x = OS_model.Monolayer()
    assert x.mu == 50

    x.set_mu(16)
    assert x.mu == 16


def test_set_time_step():
    """
    Test the function `Monolayer.set_time_step` in monolayer.py
    """
    x = OS_model.Monolayer()
    assert x.time_step == 0.005

    x.set_time_step(10)
    assert x.time_step == 10


def test_set_lam():
    """
    Test the function `Monolayer.set_lam` in monolayer.py
    """
    x = OS_model.Monolayer()
    assert x.lam == 0.1

    x.set_lam(1)
    assert x.lam == 1


def test_set_k_c():
    """
    Test the function `Monolayer.set_k_c` in monolayer.py
    """
    x = OS_model.Monolayer()
    assert x.k_c == 5

    x.set_k_c(16)
    assert x.k_c == 16


def test_set_k_pert():
    """
    Test the function `Monolayer.set_k_pert` in monolayer.py
    """
    x = OS_model.Monolayer()
    assert x.k_pert == 1

    x.set_k_pert(16)
    assert x.k_pert == 16


def test_set_drag():
    """
    Test the function `Monolayer.set_drag` in monolayer.py
    """
    x = OS_model.Monolayer()
    assert x.drag == 1

    x.set_drag(16)
    assert x.drag == 16


def test_set_r_max():
    """
        Test the function `Monolayer.set_r_max` in monolayer.py
        """
    x = OS_model.Monolayer()
    assert x.r_max == 2.5

    x.set_r_max(1)
    assert x.r_max == 1


def test_set_mag():
    """
    Test the function `Monolayer.set_mag` in monolayer.py
    """
    x = OS_model.Monolayer()
    assert x.mag == 0.05

    x.set_mag(0.2)
    assert x.mag == 0.2


def test_set_random_cells():
    """
    Test the functionality of producing random initial cells for the monolayer, via `set_random_cells` in utils.py
    """
    random.seed(1234)
    x = OS_model.Monolayer(rand=True, n=2)
    assert x.initial_positions == ((8.276, 4.401), (6.06, 2.932))


def test_neighbours():
    """
    Test the function `Monolayer.neighbours` in monolayer.py acts as expected
    """
    x = OS_model.Monolayer(size=5)
    x.manual_cell_placement(((0.0, 0.0), (4.0, 3.0), (6.0, 6.0)), [0, 0, 0])
    assert x.neighbours(5) == [[0, 1], [0, 1, 2], [1, 2]]


def test_simulation_times():
    """
    Test the functions `Monolayer.simulate_step`, 'Monolayer.simulate', 'Monolayer.measure_sorting' in monolayer.py.
    Checks that the simulation timings are correctly implemented.
    """
    x = OS_model.Monolayer()
    x.simulate_step()
    assert round(x.sim_time, 4) == x.time_step

    x.simulate(end_time=0)
    assert x.sim_time == 0

    x.simulate(end_time=0.1)
    assert round(x.sim_time, 4) == 0.1

    x.simulate(end_time=0.01)
    assert round(x.sim_time, 4) == 0.01

    _ = x.measure_sorting(end_time=0.1)
    assert round(x.sim_time, 4) == 0.1

    _ = x.measure_sorting(end_time=0)
    assert round(x.sim_time, 4) == 0


def test_reset():
    """
    Tests the function `Monolayer.reset` in monolayer.py returns cells to their initial state
    """
    random.seed(1234)
    x = OS_model.Monolayer(rand=True)
    x.simulate(1)
    x.reset()
    assert x.sim_time == 0
    for i in range(2):
        for j in range(2):
            assert x.positions[i][j] == x.initial_positions[i][j]


def test_manual_cell_placement():
    """
    Test the function `Monolayer.manual_cell_placement` in monolayer.py places cells as expected,
    and updates class properties
    """
    x = OS_model.Monolayer(size=5)
    x.manual_cell_placement(((2.0, 2.0), (2.0, 3.0)), [0, 1])
    assert x.initial_positions == ((2., 2.), (2., 3.))
    assert x.num_cells == 2
    assert np.array_equal(x.cell_types, np.asarray([0, 1]))


def test_natural_separation():
    """
    Test the simulations have the correct expected behaviour when we have two cells of the same type.
    Started from their natural separation, we expect them to stay close. Similarly for slight deviations.
    """
    base = OS_model.Monolayer(size=5)
    base.set_k_pert(0)
    base.manual_cell_placement(((2.0, 2.0), (2.0, 3.0)), [0, 0])
    apart = OS_model.Monolayer(size=5)
    apart.set_k_pert(0)
    apart.manual_cell_placement(((2.0, 1.8), (2.0, 3.0)), [0, 0])
    overlap = OS_model.Monolayer(size=5)
    overlap.set_k_pert(0)
    overlap.manual_cell_placement(((2.0, 2.2), (2.0, 3.0)), [0, 0])
    dist = np.zeros((21, 3))
    for i in range(21):
        base.simulate(10 * i)
        dist[i, 0] = np.linalg.norm(base.positions[0] - base.positions[1])
        apart.simulate(10 * i)
        dist[i, 1] = np.linalg.norm(apart.positions[0] - apart.positions[1])
        overlap.simulate(10 * i)
        dist[i, 2] = np.linalg.norm(overlap.positions[0] - overlap.positions[1])
    base_average_distance = sum(dist[:, 0]) / 21
    apart_average_distance = sum(dist[:, 1]) / 21
    overlap_average_distance = sum(dist[:, 2]) / 21
    error = 0.01
    assert abs(base_average_distance - 1) <= error
    assert abs(apart_average_distance - 1) <= error
    assert abs(overlap_average_distance - 1) <= error


def test_manual_division_timer():
    """
    Tests the function `Monolayer.manual_division_timer` in monolayer.py sets the correct division times and rates
    """
    x = OS_model.Monolayer(size=5)
    x.manual_cell_placement(([2, 2], [4, 4]), [1, 0])
    x.manual_division_timer((5, 1), (7, 2))
    assert np.array_equal(x.division_timer, np.asarray([5, 1]))
    assert np.array_equal(x.division_rates, np.asarray([7, 2]))


def test_division():
    x = OS_model.Monolayer(size=5)
    x.manual_cell_placement(([2, 2], [4, 4]), [1, 0])
    x.manual_division_timer((0.005, 1), (1, 2))
    x.simulate(end_time=0.005)
    assert np.array_equal(x.division_timer, np.asarray([0, 0.995]))
    x.simulate(end_time=0.01)
    assert x.num_cells == 3 and len(x.positions) == 3 and x.type_1 == 2 and x.type_0 == 1 and len(x.division_timer) == 3
    assert np.array_equal(x.division_rates, np.asarray([1, 2, 1]))


def test_set_division_timer():
    x = OS_model.Monolayer(size=5)
    x.manual_cell_placement(([2, 2], [4, 4]), [1, 0])
    assert x.division_timer is None
    x.set_division_timer(division_rate=13)
    assert len(x.division_timer) == 2
    assert np.array_equal(x.division_rates, np.asarray([13, 13]))
    x.set_division_timer(11, 15)
    assert np.array_equal(x.division_rates, np.asarray([15, 11]))
    assert len(x.division_timer) == 2


def test_spacing():
    x = OS_model.Monolayer(size=1)
    y = OS_model.Monolayer(space=True)
    assert x.space == 0 and y.space == 4
    x.set_space()
    assert x.space == 4


def test_boundary_condition():
    """
    Test the simulations have the correct expected behaviour when at the boundary.
    """
    for i in range(10):
        random.seed(143 * i)
        x = OS_model.Monolayer(size=1)
        x.set_k_pert(100)
        scale = sqrt(2 * x.k_pert * x.mag / x.time_step)
        x.simulate_step()
        random.seed(143 * i)
        f = OS_model.random_forces(scale, x.positions.shape)
        z = np.array([0.5, 0.5]) + (x.time_step / x.drag) * f
        for j in range(2):
            if z[0][j] < 0:
                assert x.positions[0][j] == -z[0][j]
            elif z[0][j] > 1:
                assert x.positions[0][j] == 2 - z[0][j]
            else:
                assert x.positions[0][j] == z[0][j]
