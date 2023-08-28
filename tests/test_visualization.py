import os
from mock import patch
from otsensitivity import plot_indices, pairplot


@patch("matplotlib.pyplot.show")
def test_indices_aggregated(mock_show, tmp):
    indices = [[0.314, 0.442, 0.0], [0.558, 0.442, 0.244]]

    plot_indices(indices, conf=0.05)
    plot_indices(
        indices, plabels=["x1", "t", "y"], fname=os.path.join(tmp, "sobol.pdf")
    )
    plot_indices(indices, polar=True, conf=[[0.2, 0.1, 0.1], [0.1, 0.1, 0.1]])
    plot_indices([indices[0]])


@patch("matplotlib.pyplot.show")
def test_indices_map(mock_show, tmp):
    s_first_full = [
        [0.11, 0.89],
        [0.18, 0.82],
        [0.23, 0.78],
        [0.23, 0.77],
        [0.40, 0.61],
        [0.77, 0.24],
        [0.87, 0.14],
        [0.87, 0.14],
        [0.87, 0.14],
        [0.87, 0.14],
        [0.85, 0.16],
        [0.81, 0.21],
        [0.73, 0.29],
        [0.00, 0.96],
    ]
    s_total_full = [
        [0.11, 0.87],
        [0.19, 0.79],
        [0.25, 0.74],
        [0.26, 0.74],
        [0.43, 0.57],
        [0.80, 0.22],
        [0.90, 0.11],
        [0.90, 0.11],
        [0.90, 0.11],
        [0.89, 0.11],
        [0.88, 0.136],
        [0.84, 0.178],
        [0.76, 0.25],
        [0.00, 0.98],
    ]
    s_first = [0.75, 0.21]
    s_total = [0.76, 0.24]
    xdata = [
        13150,
        19450,
        21825,
        21925,
        25775,
        32000,
        36131.67,
        36240,
        36290,
        38230.45,
        44557.5,
        51053.33,
        57550,
        62175,
    ]

    indices = [s_first, s_total, s_first_full, s_total_full]

    plot_indices(indices)
    plot_indices(
        indices,
        plabels=["Ks", "Q"],
        xdata=xdata,
        fname=os.path.join(tmp, "sobol_map.pdf"),
    )


def test_pairplot(ishigami, tmp):
    model, sample, data = ishigami
    pairplot(
        sample,
        data,
        plabels=["x1", "x2", "x3"],
        fname=os.path.join(tmp, "pairplot.pdf"),
    )
