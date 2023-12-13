import numpy as np
import argparse
from numpy import linalg as LA
from matplotlib import pyplot as plt
from scipy import ndimage
from util import Unit
from util import populate_plt_settings, get_column_width, get_text_width, get_fig_size, get_latex_float

from matplotlib import pyplot as plt
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
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
parser.add_argument("-m", "--metrics", nargs='+', default=["eulerian_masked"])
parser.add_argument("-o", "--output-prefix", type=str, required=True)
parser.add_argument("-f", "--filter", type=int, default=30)
args = parser.parse_args()

populate_plt_settings(plt)


def annotate_duration(ax,
                      text,
                      xmin,
                      xmax,
                      y,
                      texty,
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

    xcenter = xmin + (xmax - xmin) / 2
    if texty == 0:
        texty = y + (ax.get_ylim()[1] - ax.get_ylim()[0]) / 20

    ax.annotate(text, xy=(xcenter, texty), ha='center', va='center')


def plot_score(recon_dir, output_prefix, filter_size, error_histories,
               baseline_histories, metrics):
    error_filtered_list = [
        ndimage.uniform_filter(error_history, size=filter_size, mode='mirror')
        for error_history in error_histories
    ]
    baseline_filtered_list = [
        ndimage.uniform_filter(baseline_history,
                               size=filter_size,
                               mode='mirror')
        for baseline_history in baseline_histories
    ]
    score_filtered_list = [
        1 - error_filtered / baseline_filtered for error_filtered,
        baseline_filtered in zip(error_filtered_list, baseline_filtered_list)
    ]

    num_rows = 1
    num_cols = 1
    matplotlib.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(num_rows,
                           num_cols,
                           figsize=get_fig_size(get_text_width(), ratio=0.23),
                           dpi=800)
    cmap = plt.get_cmap("tab10")

    metric_names = {
        'eulerian_masked': 'Eulerian',
        'height_field': 'Height field'
    }
    metric_color_id = {'eulerian_masked': 0, 'height_field': 1}
    cmap = plt.get_cmap("tab10")
    sim_freq = 100
    ts = np.arange(len(score_filtered_list[0])) / sim_freq
    for metric_id, score_filtered in enumerate(score_filtered_list):
        metric = metrics[metric_id]
        ax.plot(ts,
                score_filtered,
                color=cmap(metric_color_id[metric]),
                label=metric_names[metric],
                zorder=3)
    frame_interval = 0.5
    for frame_id in range(3, 20, 2):
        recon_hf1 = matplotlib.image.imread(
            f"{recon_dir}/combined-hf{frame_id}.png")
        imagebox = OffsetImage(recon_hf1, zoom=0.064, alpha=0.9)
        t = frame_interval * frame_id
        annotation_bbox = AnnotationBbox(imagebox, (t, 0.5), frameon=False)
        annotation_bbox.set_zorder(2)
        ax.add_artist(annotation_bbox)
        ax.annotate(r"$\SI{" + str(t) + r"}{\second}$",
                    xy=(t, 0.943),
                    xycoords="data",
                    va="center",
                    ha="center",
                    bbox=dict(boxstyle="round,pad=0",
                              facecolor="white",
                              edgecolor='None',
                              alpha=0.6))
    ax.set_xlabel(r'$t (\SI{}{\second})$')
    ax.xaxis.set_label_coords(0.984, -0.025)
    ax.set_ylabel(r'Score')
    ax.set_ylim(0, 1)
    if len(metrics) == 1:
        ax.set_ylabel(metric_names[metrics[0]], rotation='horizontal')
        ax.yaxis.set_label_coords(0, 1.04)
    ax.set_xlim(0.833, 10)

    if output_prefix == 'diagonal-time-score':
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
                          texty=0.05)
    elif output_prefix == 'bidir-circles-time-score':
        annotate_duration(ax,
                          "Anticlockwise",
                          xmin=0.833,
                          xmax=2.08,
                          y=0,
                          texty=0.05,
                          disable_start_glyph=True)
        annotate_duration(ax,
                          "Clockwise",
                          xmin=2.08,
                          xmax=4.17,
                          y=0,
                          texty=0.05)
        annotate_duration(ax,
                          "Anticlockwise",
                          xmin=4.17,
                          xmax=6.25,
                          y=0,
                          texty=0.05)
        annotate_duration(ax,
                          "Clockwise",
                          xmin=6.25,
                          xmax=8.33,
                          y=0,
                          texty=0.05)
        annotate_duration(ax,
                          "Anticlockwise",
                          xmin=8.33,
                          xmax=10,
                          y=0,
                          texty=0.05)

    epsilon = 0.01
    ax.xaxis.set_ticks(np.arange(1, 9.5 + epsilon, 0.5))
    ax.yaxis.set_ticks(np.arange(0, 1 + epsilon, 0.1))

    if output_prefix == 'diagonal-time-score':
        ax.legend(loc='upper center',
                  bbox_to_anchor=(0.92, 0.24),
                  labelspacing=0.3,
                  borderpad=0)
    else:
        ax.legend(loc='upper center',
                  bbox_to_anchor=(0.593, 0.24),
                  labelspacing=0.3,
                  borderpad=0)
    fig.tight_layout(pad=0.07)
    fig.savefig(f'{output_prefix}.pdf')
    plt.close('all')


metrics = args.metrics
recon_dir = args.recon_directory

error_histories = [
    np.load(f'{recon_dir}/{metric}_error.npy') for metric in metrics
]
baseline_histories = [
    np.load(f'{recon_dir}/{metric}_baseline.npy') for metric in metrics
]
num_samples_histories = [
    np.load(f'{recon_dir}/{metric}_num_samples.npy') for metric in metrics
]

baseline_accm = np.zeros(len(metrics))
error_accm = np.zeros(len(metrics))

for metric_id in range(len(metrics)):
    for frame_id in range(len(error_histories[metric_id])):
        num_samples = num_samples_histories[metric_id][frame_id]
        if num_samples > 0:
            baseline_accm[metric_id] += baseline_histories[metric_id][
                frame_id] / num_samples
            error_accm[metric_id] += error_histories[metric_id][
                frame_id] / num_samples

plot_score(recon_dir, args.output_prefix, args.filter, error_histories,
           baseline_histories, metrics)
