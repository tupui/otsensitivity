import pytest
import numpy as np
from otsensitivity import sobol_saltelli


def test_sobol(ishigami):
    model, sample, data = ishigami

    s, st = sobol_saltelli(
        model, 1000, 3, [[-np.pi, -np.pi, -np.pi], [np.pi, np.pi, np.pi]]
    )

    assert s == pytest.approx(np.array([0.314, 0.442, 0.0]), abs=0.1)
    assert st == pytest.approx(np.array([0.558, 0.442, 0.244]), abs=0.1)
