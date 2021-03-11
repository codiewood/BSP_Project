import OS_model
import numpy as np
from numpy import random
from scipy.spatial.distance import euclidean


def test_uniform_coords():
    """
    Test the function `uniform_coords` in main.py
    """
    random.seed(1234)
    assert OS_model.uniform_coords(20) == (16.361, 8.18)


def test_set_mu():
    """
    Test the function `Monolayer.set_mu` in main.py
    """
    x = OS_model.Monolayer(1)
    assert x.mu == 50

    x.set_mu(16)
    assert x.mu == 16


def test_set_lam():
    """
    Test the function `Monolayer.set_lam` in main.py
    """
    x = OS_model.Monolayer(1)
    assert x.lam == 0.1

    x.set_lam(1)
    assert x.lam == 1


def test_set_k_c():
    """
    Test the function `Monolayer.set_k_c` in main.py
    """
    x = OS_model.Monolayer(1)
    assert x.k_c == 5

    x.set_k_c(16)
    assert x.k_c == 16


def test_set_k_pert():
    """
    Test the function `Monolayer.set_k_pert` in main.py
    """
    x = OS_model.Monolayer(1)
    assert x.k_pert == 1

    x.set_k_pert(16)
    assert x.k_pert == 16


def test_simulation_parameters():
    """
    Test the function `Monolayer.simulation_parameters` in main.py
    """
    x = OS_model.Monolayer(1)
    assert x.sim_params[0] == 2.5 and x.sim_params[1] == 0.05 and x.sim_params[2] == 1

    x.simulation_parameters(1, 2, 3)
    assert x.sim_params[0] == 1 and x.sim_params[1] == 2 and x.sim_params[2] == 3


def test_set_random_cells():
    """
    Test the function `Monolayer.set_random_cells` in main.py
    """
    random.seed(1234)
    x = OS_model.Monolayer(rand=True, n=2)
    assert x.initial_positions == ((8.276, 4.401), (6.06, 2.932))


def test_generate_initial_positions_array():
    """
    Test the function `Monolayer.generate_initial_positions_array` in main.py
    """
    random.seed(1234)
    x = OS_model.Monolayer(size=5)
    for i in range(2):
        for j in range(2):
            assert x.generate_initial_positions_array()[i][j] == x.initial_positions[i][j]
    assert isinstance(x.generate_initial_positions_array(), np.ndarray)


def test_neighbours():
    """
    Test the function `Monolayer.neighbours` in main.py
    """


def test_interaction_forces():
    """
    Test the function `Monolayer.interaction_forces` in main.py
    """


def test_simulation_times():
    """
    Test the functions `Monolayer.simulate_step` and 'Monolayer.simulate' in main.py.
    Checks that the simulation timings are correctly implemented.
    """
    random.seed(1234)
    x = OS_model.Monolayer(size=5)
    x.simulate_step(time_step=0.005)
    assert round(x.sim_time, 3) == 0.005
    x.simulate(end_time=0)
    assert x.sim_time == 0
    x.simulate(end_time=0.1)
    assert round(x.sim_time, 3) == 0.1
    x.simulate(end_time=0.01)
    assert round(x.sim_time, 3) == 0.01

def test_reset():
    """
    Test the function `Monolayer.reset` in main.py
    """
    random.seed(1234)
    x = OS_model.Monolayer(rand=True)
    x.simulate(1)
    x.reset()
    assert x.sim_time == 0
    for i in range(2):
        for j in range(2):
            assert x.positions[i][j] == x.initial_positions[i][j]


def test_simulate():
    """
    Test the function `Monolayer.simulate` in main.py
    """


def test_manual_cell_placement():
    """
    Test the function `Monolayer.manual_cell_placement` in main.py
    """
    x = OS_model.Monolayer(size=5)
    x.manual_cell_placement(((2.0, 2.0), (2.0, 3.0)), [0, 1])
    assert x.initial_positions == ((2., 2.), (2., 3.))
    assert x.num_cells == 2
    assert x.cell_types == [0, 1]


def test_natural_separation():
    """
    Test the simulations have the correct expected behaviour when we have two cells of the same type.
    Started from their natural separation, we expect them to stay close. Similarly for slight deviations.
    """
    random.seed(1234)
    base = OS_model.Monolayer(size=5)
    base.manual_cell_placement(((2.0, 2.0), (2.0, 3.0)), [0, 0])
    apart = OS_model.Monolayer(size=5)
    apart.manual_cell_placement(((2.0, 1.8), (2.0, 3.0)), [0, 0])
    overlap = OS_model.Monolayer(size=5)
    overlap.manual_cell_placement(((2.0, 2.2), (2.0, 3.0)), [0, 0])
    dist = np.zeros((21, 3))
    for i in range(21):
        base.simulate(10 * i)
        dist[i, 0] = euclidean(base.positions[0], base.positions[1])
        apart.simulate(10 * i)
        dist[i, 1] = euclidean(apart.positions[0], apart.positions[1])
        overlap.simulate(10 * i)
        dist[i, 2] = euclidean(overlap.positions[0], overlap.positions[1])
    base_average_distance = sum(dist[:, 0]) / 21
    apart_average_distance = sum(dist[:, 1]) / 21
    overlap_average_distance = sum(dist[:, 2]) / 21
    error = 0.025
    assert abs(base_average_distance - 1) <= error
    assert abs(apart_average_distance - 1) <= error
    assert abs(overlap_average_distance - 1) <= error
