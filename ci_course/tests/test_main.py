import OS_model as bsp
import numpy as np


def test_rand_pert():
    """
    Test the function `rand_pert` in main.py
    """
    np.random.seed(1234)
    assert np.round(bsp.rand_pert(1, 1), 5) == np.array([0.66671])

def test_uniform_coords():
    """
    Test the function `uniform_coords` in main.py
    """
    np.random.seed(1234)
    assert bsp.uniform_coords(20) == (16.361, 8.18)


def test_set_mu():
    """
    Test the function `Monlayer.set_mu` in main.py
    """
    x = bsp.Monolayer(1)

    assert x.mu == 50 and x.lam == 0.1

    x.set_mu(5)
    assert x.mu == 5 and x.lam == 0.1

    x.set_mu(lam=1)
    assert x.mu == 5 and x.lam == 1

def test_set_k_c():
    """
    Test the function `Monlayer.set_k_c` in main.py
    """
    x = bsp.Monolayer(1)
    assert x.k_c == 5

    x.set_k_c(16)
    assert x.mu == 16

def test_set_radius():
    """
    Test the function `Monlayer.set_radius` in main.py
    """
    x = bsp.Monolayer(1)
    assert x.r_0 == 1 and x.r_1 == 1

    x.set_radius(16)
    assert x.r_0 == 1 and x.r_1 == 16
