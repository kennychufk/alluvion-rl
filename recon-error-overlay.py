import argparse
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy import ndimage
from scipy import interpolate

matplotlib.use('Agg')

parser = argparse.ArgumentParser()
parser.add_argument("-r",
                    "--recon-directory",
                    type=str,
                    required=True,
                    help="reconstruction directory")
parser.add_argument("-m", "--metrics", nargs='+', default=["eulerian_masked"])
parser.add_argument("-i", "--input-prefix", type=str, default="beads")
parser.add_argument("-f", "--filter", type=int, default=30)
args = parser.parse_args()


def render_graph(recon_dir, frame_id, input_prefix, filter_size,
                 error_histories, baseline_histories, metrics):
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

    fps = 30
    t = frame_id / fps
    my_dpi = 60
    img_filename = f"{recon_dir}/{input_prefix}{frame_id}.png"

    fig = plt.figure(figsize=[1920 // my_dpi, 1080 // my_dpi],
                     dpi=my_dpi,
                     frameon=False)
    img_ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(img_ax)
    img = plt.imread(img_filename)

    img_ax.imshow(img, cmap='gray')
    img_ax.set_axis_off()

    if len(metrics) > 1:
        ax_box = fig.add_axes([0.327, 0.586, 0.333, 0.374])
    else:
        ax_box = fig.add_axes([0.312, 0.591, 0.348, 0.379])
    ax_box.xaxis.set_visible(False)
    ax_box.yaxis.set_visible(False)
    #     ax_box.set_zorder(1000)
    ax_box.patch.set_alpha(0.5)
    ax_box.patch.set_color('white')

    metric_names = {
        'eulerian_masked': 'Eulerian score',
        'height_field': 'Height field score'
    }
    metric_color_id = {'eulerian_masked': 0, 'height_field': 1}
    ax_score = fig.add_axes([0.35, 0.638, 0.3, 0.3])
    cmap = plt.get_cmap("tab10")
    sim_freq = 100
    ts = np.arange(len(score_filtered_list[0])) / sim_freq
    for metric_id, score_filtered in enumerate(score_filtered_list):
        metric = metrics[metric_id]
        ax_score.plot(ts,
                      score_filtered,
                      color=cmap(metric_color_id[metric]),
                      label=metric_names[metric])
    ax_score.set_xlabel(r'$t$ (s)')
    ax_score.set_ylim(0, 1)
    if len(metrics) == 1:
        ax_score.set_ylabel(metric_names[metrics[0]], rotation='horizontal')
        ax_score.yaxis.set_label_coords(0, 1.04)
    ax_score.set_xlim(0, 10)

    epsilon = 0.01
    ax_score.xaxis.set_ticks(np.arange(0, 10 + epsilon, 1))
    ax_score.yaxis.set_ticks(np.arange(0, 1 + epsilon, 0.1))

    if t < 10:
        ax_score.axvline(x=t, color=cmap(2))
    for metric_id, score_filtered in enumerate(score_filtered_list):
        score_interp = interpolate.interp1d(ts, score_filtered)
        if t < 10:
            inst_score = score_interp(t)
            if inst_score > -1:
                # ax_score.plot((0, t), (inst_score, inst_score), color=cmap(2))
                metric = metrics[metric_id]
                ax_score.plot((0, t), (inst_score, inst_score),
                              color=cmap(metric_color_id[metric]))

    ax_score.patch.set_alpha(0.8)
    if len(metrics) > 1:
        ax_score.legend(loc='lower right')

    fig.savefig(f'{recon_dir}/{input_prefix}-graph-{frame_id}.png', dpi=my_dpi)
    #     [left, bottom, width, height]

    plt.close('all')


plt.rcParams.update({'font.size': 22})

recon_dir = args.recon_directory

metrics = args.metrics
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
            error_accm[metric_id] += error_histories[metric_id][frame_id] / num_samples
        # print(num_samples, baseline_accm, error_accm)

print('Score', error_accm / baseline_accm)

for frame_id in range(300):
    render_graph(recon_dir, frame_id, args.input_prefix, args.filter,
                 error_histories, baseline_histories, metrics)
