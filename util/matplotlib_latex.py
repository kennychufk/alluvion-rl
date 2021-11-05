def populate_plt_settings(plt):
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",  # use serif/main font for text elements
        "text.usetex": True,  # use inline math for ticks
        "pgf.rcfonts": False,  # don't setup fonts from rc parameters
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 8,
        "font.size": 8,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8
    })
    plt.style.use('seaborn-whitegrid')


def get_column_width():
    return 243.14749  #pt. # \showthe\columnwidth


def get_text_width():
    return 510.295  #pt. # \showthe\textwidth


def get_fig_size(width, fraction=1, ratio=None):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2
    if ratio is None:
        ratio = golden_ratio

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def get_latex_float(f):
    float_str = "{0:.3g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str
