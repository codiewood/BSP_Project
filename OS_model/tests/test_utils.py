import OS_model
import numpy as np
from numpy import random


def test_uniform_coords():
    """
    Test the function `uniform_coords` in utils.py
    """
    random.seed(1234)
    assert OS_model.uniform_coords(20) == (16.361, 8.18)


def test_random_unit_vector():
    """
    Test the function `random_unit_vector` in utils.py returns a unit vector
    """
    x = OS_model.random_unit_vector()
    assert round(np.linalg.norm(x), 5) == 1


def test_generate_initial_positions_array():
    """
    Test the function `generate_initial_positions_array` in utils.py
    """
    positions = ((1, 2), (4, 15), (6.2, 3.6))
    x = OS_model.generate_initial_positions_array(positions)
    for i in range(2):
        for j in range(2):
            assert x[i][j] == positions[i][j]
    assert isinstance(x, np.ndarray)
    assert x.shape == (3, 2)
