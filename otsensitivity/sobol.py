# coding: utf8
import numpy as np


def sobol_saltelli(func, n_sample, dim, bounds=None):
    """Sobol' indices using formulation from Saltelli2010.

    The total number of function call is N(p+2).
    Three matrices are required for the computation of
    the indices: A, B and a permutation matrix AB based
    on both A and B.

    References
    ----------
    [1] Saltelli et al. Variance based sensitivity analysis of model output.
      Design and estimator for the total sensitivity index, Computer Physics
      Communications, 2010. DOI: 10.1016/j.cpc.2009.09.018

    :param callable func: Function to analyse.
    :param int n_sample: Number of samples.
    :param int dim: Number of dimensions.
    :param array_like bounds: Desired range of transformed data.
      The transformation apply the bounds on the sample and not
      the theoretical space, unit cube. Thus min and
      max values of the sample will coincide with the bounds.
      ([min, n_features], [max, n_features]).
    :return: first orders and total orders indices.
    :rtype: list(float) (n_features), list(float) (n_features)
    """
    A = np.random.random_sample((n_sample, dim))
    B = np.random.random_sample((n_sample, dim))

    if bounds is not None:
        bounds = np.asarray(bounds)
        min_ = bounds.min(axis=0)
        max_ = bounds.max(axis=0)

        A = (max_ - min_) * A + min_
        B = (max_ - min_) * B + min_

    f_A = np.asarray(func(A))
    f_B = np.asarray(func(B))

    var = np.var(np.vstack([f_A, f_B]), axis=0)

    # Total effect of pairs of factors: generalization of Saltenis
    # st2 = np.zeros((dim, dim))
    # for j, i in itertools.combinations(range(0, dim), 2):
    #     st2[i, j] = 1 / (2 * n_sample) * np.sum((f_AB[i] - f_AB[j]) ** 2, axis=0) / var

    # Saltenis formulation (Jansen)
    # f_AB = []
    # s = []
    # st = []
    # for i in range(dim):
    #     f_AB.append(func(np.column_stack((A[:, 0:i], B[:, i], A[:, i+1:]))))
    #     s.append(1 / n_sample * np.sum(f_B * (f_AB[i] - f_A), axis=0) / var)
    #     st.append(1 / (2 * n_sample) * np.sum((f_A - f_AB[i]) ** 2, axis=0) / var)

    f_AB = []
    for i in range(dim):
        f_AB.append(func(np.column_stack((A[:, 0:i], B[:, i], A[:, i + 1 :]))))

    f_AB = np.asarray(f_AB).reshape(dim, n_sample)

    s = 1 / n_sample * np.sum(f_B * (np.subtract(f_AB, f_A.flatten()).T), axis=0) / var
    st = (
        1
        / (2 * n_sample)
        * np.sum(np.subtract(f_A.flatten(), f_AB).T ** 2, axis=0)
        / var
    )

    return s, st
