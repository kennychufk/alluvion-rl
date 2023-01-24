preamble_str = r"""
% >>> from acmart.cls
% https://github.com/borisveytsman/acmart/blob/primary/acmart.dtx#L3306
\usepackage[T1]{fontenc}
\usepackage[tt=false, type1=true]{libertine}
\usepackage[varqu]{zi4}
\usepackage[libertine]{newtxmath}
% <<< from acmart.cls

% default font size for acmtog
\usepackage[fontsize=9pt]{fontsize}

\usepackage{physics} % For mathematical symbols
\usepackage{amsmath} % For multiple equations at the same block
\usepackage{siunitx}

\newcommand{\xvec}{\mathbf{x}}
\newcommand{\vvec}{\mathbf{v}}
\newcommand{\avec}{\mathbf{a}}
\newcommand{\fvec}{\mathbf{F}}
\newcommand{\vort}{\boldsymbol{\mathbf{\omega}}}
\newcommand{\quat}{\mathbf{q}}
\newcommand{\xnew}{\xvec^{\mathrm{new}}}
\newcommand{\vorticity}{\mathbf{\omega}}
\newcommand{\vorticitylocation}{\mathbf{\eta}}
\newcommand{\kernel}[2]{W\left(#1, #2\right)}
\newcommand{\kernelnamed}[3]{W_{#1}\left(#2, #3\right)}
\newcommand{\gradkernel}[2]{\nabla W\left(#1, #2\right)}
\newcommand{\gradkernelnamed}[3]{\nabla W_{#1}\left(#2, #3\right)}
\newcommand{\dx}[1]{\frac{\partial #1}{\partial x}}
\newcommand{\dy}[1]{\frac{\partial #1}{\partial y}}
\newcommand{\dxx}[1]{\frac{\partial^2 #1}{\partial x^2}}
\newcommand{\dyy}[1]{\frac{\partial^2 #1}{\partial y^2}}
\newcommand{\dxy}[1]{\frac{\partial^2 #1}{\partial x \partial y}}
\newcommand{\condition}[2]{\left. #1 \right\rvert_{#2}}
\newcommand\Rey{\mbox{\textit{Re}}} 
\DeclareSIUnit\stokes{St}
"""

def populate_plt_settings(plt):
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",  # use serif/main font for text elements
        "text.usetex": True,  # use inline math for ticks
        "text.latex.preamble":  preamble_str,
        "pgf.rcfonts": False,  # don't setup fonts from rc parameters
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 8,
        "font.size": 8,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.titlesize": 8,
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
