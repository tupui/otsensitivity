import os
import numpy as np
import numpy.testing as npt
from otsensitivity import cusunoro, ecdf, moment_independent


def test_cusunoro(ishigami, tmp):
    model, X, Y = ishigami
    cuso = cusunoro(
        X, Y, plabels=["x1", "x2", "x3"], fname=os.path.join(tmp, "cusunoro.pdf")
    )

    npt.assert_almost_equal(cuso[2], [0.326, 0.448, 0.004], decimal=3)


def test_ecdf():
    data = np.array([1, 3, 6, 10, 2])
    xs, ys = ecdf(data)
    npt.assert_equal(xs, [1, 2, 3, 6, 10])
    npt.assert_equal(ys, [0, 0.25, 0.5, 0.75, 1.0])


def test_moment_independant(ishigami, tmp):
    model, X, Y = ishigami

    momi = moment_independent(
        X,
        Y,
        plabels=["x1", "x2", "x3"],
        fname=os.path.join(tmp, "moment_independent.pdf"),
    )

    npt.assert_almost_equal(momi[2]["Kolmogorov"], [0.21, 0.3, 0.1], decimal=2)
    npt.assert_almost_equal(momi[2]["Kuiper"], [0.22, 0.3, 0.19], decimal=2)
    npt.assert_almost_equal(momi[2]["Delta"], [0.2, 0.27, 0.17], decimal=2)
    npt.assert_almost_equal(momi[2]["Sobol"], [0.29, 0.29, 0.00], decimal=2)

    # Cramer
    X = np.random.normal(0, 1, size=[5000, 2])
    Y = [np.exp(x_i[0] + 2 * x_i[1]) for x_i in X]

    momi = moment_independent(X, Y, fname=os.path.join(tmp, "moment_independent.pdf"))

    npt.assert_almost_equal(momi[2]["Cramer"], [0.11, 0.57], decimal=2)
