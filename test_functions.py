import numpy as np
from cond_posteriors import vbar


def test_vbar():
    X = np.array([[1, 1, 1], [1, 0, -1], [1, 1, 1], [1, 0, 1]])
    assert vbar(X) == 1 / 3


test_vbar()