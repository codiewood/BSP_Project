import OS_model
import numpy as np
from numpy import random


def test_rand_pert():
    """
    Test the function `rand_pert` in main.py
    """
    random.seed(1234)
    assert np.round(OS_model.rand_pert(1, 1), 5) == np.array([0.66671])


def test_uniform_coords():
    """
    Test the function `uniform_coords` in main.py
    """
    np.random.seed(1234)
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


def test_set_radius():
    """
    Test the function `Monolayer.set_radius` in main.py
    """
    x = OS_model.Monolayer(1)
    assert x.r0 == 1 and x.r1 == 1

    x.set_radius(16)
    assert x.r0 == 1 and x.r1 == 16


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
    x = OS_model.Monolayer(2)
    assert x.initial_positions == ((16.361, 8.18), (11.683, 5.078))


def test_generate_initial_positions_array():
    """
    Test the function `Monolayer.generate_initial_positions_array` in main.py
    """
    random.seed(1234)
    x = OS_model.Monolayer(2)
    for i, j in zip(range(2), range(2)):
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


def test_simulate_step():
    """
    Test the function `Monolayer.simulate_step` in main.py
    """


def test_reset():
    """
    Test the function `Monolayer.reset` in main.py
    """
    random.seed(1234)
    x = OS_model.Monolayer(2)
    x.simulate(5)
    x.reset()
    assert x.sim_time == 0
    for i, j in zip(range(2), range(2)):
        assert x.positions[i][j] == x.initial_positions[i][j]


def test_simulate():
    """
    Test the function `Monolayer.simulate` in main.py
    """
