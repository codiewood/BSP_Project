import ci_course
import numpy as np


def test_rand_pert():
    """
    Test the function `rand_pert` in main.py
    """
    np.random.seed(1234)
    assert np.round(ci_course.rand_pert(1, 1, 1), 5) == np.array([0.66671])
    assert ci_course.rand_pert() == np.array([])
