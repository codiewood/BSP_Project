import ci_course as bsp
import numpy as np


def test_rand_pert(): # Unsure if this test is rigorous enough
    """
    Test the function `rand_pert` in main.py
    """
    np.random.seed(1234)
    assert np.round(bsp.rand_pert(1, 1), 5) == np.array([0.66671])


def test_set_mu():
    x = Monolayer(1)
    assert x.mu == 50 and x.mu_het == 5
    x.set_mu(1)
    assert x.mu == 1 and x.mu_het == 5
    x.set_mu(mu_het=7)
    assert x.mu == 5 and x.mu_het == 7


# TODO: Test for uniform coords (?)
# TODO: Tests for class functions
# TODO: Docstrings
