import numpy as np
import argparse
from numpy import linalg as LA
from matplotlib import pyplot as plt
from scipy import ndimage
from util import Unit
from util import populate_plt_settings, get_column_width, get_text_width, get_fig_size, get_latex_float

from matplotlib import pyplot as plt
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from matplotlib.patches import ArrowStyle
import matplotlib
import matplotlib.image

matplotlib.use('Agg')

parser = argparse.ArgumentParser()
parser.add_argument("-r",
                    "--recon-directory",
                    type=str,
                    required=True,
                    help="reconstruction directory")
parser.add_argument("-o", "--output-prefix", type=str, required=True)
parser.add_argument("-f", "--filter", type=int, default=200)
args = parser.parse_args()

populate_plt_settings(plt)


def annotate_duration(ax,
                      text,
                      xmin,
                      xmax,
                      y,
                      texty,
                      textx=None,
                      linecolor="black",
                      linewidth=0.5,
                      disable_start_glyph=False):
    if disable_start_glyph:
        ax.annotate('',
                    xy=(xmin, y),
                    xytext=(xmax, y),
                    xycoords='data',
                    textcoords='data',
                    arrowprops={
                        'arrowstyle': ArrowStyle.BarAB(widthA=0.3, widthB=0.0),
                        'color': linecolor,
                        'linewidth': linewidth,
                        'shrinkA': 0,
                        'shrinkB': 0
                    })
        ax.annotate('',
                    xy=(xmin, y),
                    xytext=(xmax, y),
                    xycoords='data',
                    textcoords='data',
                    arrowprops={
                        'arrowstyle': '<-',
                        'color': linecolor,
                        'linewidth': linewidth,
                        'shrinkA': 0,
                        'shrinkB': 0
                    })
    else:
        ax.annotate('',
                    xy=(xmin, y),
                    xytext=(xmax, y),
                    xycoords='data',
                    textcoords='data',
                    arrowprops={
                        'arrowstyle': ArrowStyle.BarAB(widthA=0.3, widthB=0.3),
                        'color': linecolor,
                        'linewidth': linewidth,
                        'shrinkA': 0,
                        'shrinkB': 0
                    })
        ax.annotate('',
                    xy=(xmin, y),
                    xytext=(xmax, y),
                    xycoords='data',
                    textcoords='data',
                    arrowprops={
                        'arrowstyle': '<->',
                        'color': linecolor,
                        'linewidth': linewidth,
                        'shrinkA': 0,
                        'shrinkB': 0
                    })

    if textx is None:
        textx = xmin + (xmax - xmin) / 2
    if texty == 0:
        texty = y + (ax.get_ylim()[1] - ax.get_ylim()[0]) / 20

    ax.annotate(text, xy=(textx, texty), ha='center', va='center')


def plot_score(recon_dir, output_prefix, filter_size, error_history,
               baseline_history):
    error_filtered = ndimage.uniform_filter(error_history,
                                            size=filter_size,
                                            mode='mirror')
    baseline_filtered = ndimage.uniform_filter(baseline_history,
                                               size=filter_size,
                                               mode='mirror')
    score_filtered = 1 - error_filtered / baseline_filtered

    is_linear_circle = (output_prefix == 'piv-linear-circle-time-score')
    is_diagonal = (output_prefix == 'piv-diagonal-time-score')

    num_rows = 1
    num_cols = 1
    matplotlib.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(num_rows,
                           num_cols,
                           figsize=get_fig_size(get_text_width(), ratio=0.2),
                           dpi=600)
    cmap = plt.get_cmap("tab10")

    piv_freq = 500
    ts = np.arange(len(score_filtered)) / piv_freq
    ax.plot(ts, score_filtered, zorder=3)

    frame_interval = 0.5
    for frame_id in range(7, 36, 4):
        recon_hf1 = matplotlib.image.imread(
            f"{recon_dir}/combined-hf{frame_id}.png")
        imagebox = OffsetImage(recon_hf1, zoom=0.048, alpha=0.9)
        t = frame_interval * frame_id
        annotation_bbox = AnnotationBbox(imagebox, (t, 0.5), frameon=False)
        annotation_bbox.set_zorder(2)
        ax.add_artist(annotation_bbox)
        ax.annotate(f"{t} s",
                    xy=(t, 0.943),
                    xycoords="data",
                    va="center",
                    ha="center")

    ax.set_xlabel(r'$t$ (s)')
    ax.set_ylim(0, 1)
    ax.set_ylabel(r"2D Eulerian score")
    if is_linear_circle:
        ax.set_xlim(0, 14)
    else:
        ax.set_xlim(0.833, 19.554)

    fig.tight_layout(
        pad=0.05
    )  # pad is 1.08 by default https://stackoverflow.com/a/59252633

    if is_diagonal:
        annotate_duration(ax,
                          "Diagonal 1",
                          xmin=0.833,
                          xmax=4.85,
                          y=0,
                          texty=0.05)
        annotate_duration(ax,
                          "Diagonal 2",
                          xmin=5.083,
                          xmax=9.1,
                          y=0,
                          texty=0.05,
                          textx=8.5)
    elif is_linear_circle:
        annotate_duration(ax,
                          "Oscillation",
                          xmin=0.983,
                          xmax=6.78,
                          y=0,
                          texty=0.05)
        annotate_duration(ax, "Loop", xmin=7.15, xmax=10.9, y=0, texty=0.05)

    epsilon = 0.01
    if is_linear_circle:
        ax.xaxis.set_ticks(np.arange(0, 14 + epsilon, 1))
    else:
        ax.xaxis.set_ticks(np.arange(1, 19 + epsilon, 1))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(which='minor', color='#DDDDDD', linewidth=0.5)
    ax.yaxis.set_ticks(np.arange(0, 1 + epsilon, 0.1))
    fig.savefig(
        f'{output_prefix}.pgf', bbox_inches='tight'
    )  # bbox_inches='tight' necessary for keeping the time legend inside the canvas
    with open(f'{output_prefix}.pgf', 'rt') as f:
        text = f.read()
        text = text.replace(output_prefix, f'res/{output_prefix}')
    with open(f'{output_prefix}.pgf', 'wt') as f:
        f.write(text)


metric = 'eulerian_masked'
recon_dir = args.recon_directory
error_history = np.load(f'{recon_dir}/{metric}_error.npy')
baseline_history = np.load(f'{recon_dir}/{metric}_baseline.npy')
num_samples_history = np.load(f'{recon_dir}/{metric}_num_samples.npy')

plot_score(recon_dir, args.output_prefix, args.filter, error_history,
           baseline_history)

# linear-circle:
# recon_dir = '/media/kennychufk/old-ubuntu/evaluation-results/2v7m4mucAug-val-piv-0.011/val-0416_114327-2v7m4muc-2600-685aa05c'
