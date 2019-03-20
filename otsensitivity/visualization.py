import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf


def save_show(fname, figures, **kwargs):
    """Either show or save the figure[s].

    If :attr:`fname` is `None` the figure will show.

    :param str fname: whether to export to filename or display the figures.
    :param list(Matplotlib figure instance) figures: Figures to handle.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for fig in figures:
            try:
                fig.tight_layout()
            except ValueError:
                pass

    if fname is not None:
        pdf = matplotlib.backends.backend_pdf.PdfPages(fname)
        for fig in figures:
            pdf.savefig(fig, transparent=True, bbox_inches='tight', **kwargs)
        pdf.close()
    else:
        plt.show()
    plt.close('all')


def plot_indices(indices, conf=None, plabels=None, polar=False, xdata=None,
                 xlabel='x', fname=None):
    """Plot Sensitivity indices.

    If `len(indices)>2` map indices are also plotted along with aggregated
    indices.

    :param array_like indices: `[first (n_features), total (n_features),
        first (xdata, n_features), total (xdata, n_features)]`.
    :param float/array_like conf: relative error around indices. If float,
        same error is applied for all parameters. Otherwise shape
        (n_features, [first, total] orders).
    :param list(str) plabels: parameters' names.
    :param bool polar: Whether to use bar chart or polar bar chart.
    :param array_like xdata: 1D discretization of the function (n_features,).
    :param str xlabel: label of the discretization parameter.
    :param str fname: whether to export to filename or display the figures.
    :returns: figure.
    :rtype: Matplotlib figure instances, Matplotlib AxesSubplot instances.
    """
    indices = np.array([np.asarray(indice) for indice in indices])

    if np.isscalar(conf):
        conf = [conf, conf]
    elif conf is None:
        conf = [None, None]
    p_len = len(indices[0])
    if plabels is None:
        plabels = ['x' + str(i) for i in range(p_len)]
    objects = [[r"$S_{" + p + r"}$", r"$S_{T_{" + p + r"}}$"]
               for i, p in enumerate(plabels)]

    s_lst = [item for sublist in objects for item in sublist]
    x_pos = np.arange(p_len)

    figs = []
    axs = []

    fig, ax = plt.subplots(subplot_kw=dict(polar=polar))
    figs.append(fig)
    axs.append(ax)

    if not polar:
        if len(indices) > 1:
            # Total orders
            ax.bar(x_pos, indices[1] - indices[0], capsize=4, ecolor='g',
                   error_kw={'elinewidth': 3, 'capthick': 3}, yerr=conf[1],
                   align='center', alpha=0.5, color='c', bottom=indices[0],
                   label='Total order')

        # First orders
        ax.bar(x_pos, indices[0], capsize=3, yerr=conf[0], align='center',
               alpha=0.5, color='r', label='First order')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(plabels)
        ax.set_ylabel('Sensitivity indices')
        ax.set_xlabel('Input parameters')
    else:

        def _polar_caps(theta, radius, ax, color='k', linewidth=1):
            """Error bar caps in polar coordinates."""
            peri = np.pi * 1 / 180
            for th, _r in zip(theta, radius):
                th_err = peri / _r
                local_theta = np.linspace(-th_err / 2, th_err / 2, 10) + th
                local_r = np.ones(10) * _r
                ax.plot(local_theta, local_r, color=color, marker='',
                        linewidth=linewidth, label=None)
            return ax

        theta = np.linspace(0.0, 2 * np.pi, p_len, endpoint=False)

        ax.bar(theta, indices[0], width=2 * np.pi / p_len,
               alpha=0.5, tick_label=plabels, color='r', label='First order')

        if len(indices) > 1:
            ax.bar(theta, indices[1] - indices[0], width=2 * np.pi / p_len,
                   alpha=0.5, color='c', bottom=indices[0], ecolor='g',
                   label='Total order')

        # Separators
        maxi = np.max([indices[0], indices[1]])
        ax.plot([theta + np.pi / p_len, theta + np.pi / p_len],
                [[0] * p_len, [maxi] * p_len], c='gray', label=None)

        if conf[0] is not None:
            # Total orders errors caps
            _polar_caps(theta, indices[1] + conf[1], ax, color='g', linewidth=3)
            _polar_caps(theta, indices[1] - conf[1], ax, color='g', linewidth=3)
            rad_ = np.array([indices[1] + conf[1], indices[1] - conf[1]])
            rad_[rad_ < 0] = 0
            ax.plot([theta, theta], rad_, color='g', linewidth=3, label=None)

            # First orders errors caps
            _polar_caps(theta, indices[0] + conf[0], ax, color='k')
            _polar_caps(theta, indices[0] - conf[0], ax, color='k')
            rad_ = np.array([indices[0] + conf[0], indices[0] - conf[0]])
            rad_[rad_ < 0] = 0
            ax.plot([theta, theta], rad_, color='k', label=None)

        ax.set_rmin(0)

    ax.legend()

    if len(indices) > 2:
        n_xdata = len(indices[3])
        if xdata is None:
            xdata = np.linspace(0, 1, n_xdata)

        fig = plt.figure('Sensitivity Map')
        ax = fig.add_subplot(111)
        figs.append(fig)
        axs.append(ax)

        indices = np.hstack(indices[2:]).T
        s_lst = np.array(objects).T.flatten('C').tolist()
        for sobol, label in zip(indices, s_lst):
            ax.plot(xdata, sobol, linewidth=3, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Sensitivity ')
        ax.set_ylim(-0.1, 1.1)
        ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')

    save_show(fname, figs)

    return figs, axs
