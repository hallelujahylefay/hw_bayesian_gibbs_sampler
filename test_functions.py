import numpy as np
from cond_posteriors import vbar, Xtilde


def test_vbar():
    X = np.array([[1, 1, 1], [1, 0, -1], [1, 1, 1], [1, 0, 1]])
    assert vbar(X) == 1 / 3


test_vbar()


def test_xtilde():
    X = np.array([[1, 1, 1], [1, 0, -1], [1, 1, 1], [1, 0, 1]])
    z = np.array([1, 0, 1])
    Xtilde_v = Xtilde(X, z)
    assert Xtilde_v.shape == (4, 2)


test_xtilde()