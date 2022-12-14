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
parser.add_argument("--empirical-scene", type=int, default=0)
parser.add_argument("-m", "--metric", type=str, default="eulerian_masked")
parser.add_argument("-i", "--input-prefix", type=str, default="beads")
parser.add_argument("-f", "--filter", type=int, default=30)
args = parser.parse_args()


def render_graph(recon_dir, frame_id, input_prefix, filter_size, error_history,
                 baseline_history):
    error_filtered = ndimage.uniform_filter(error_history,
                                            size=filter_size,
                                            mode='mirror')
    baseline_filtered = ndimage.uniform_filter(baseline_history,
                                               size=filter_size,
                                               mode='mirror')
    score_filtered = 1 - error_filtered / baseline_filtered

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

    ax_box = fig.add_axes([0.305, 0.585, 0.355, 0.379])
    ax_box.xaxis.set_visible(False)
    ax_box.yaxis.set_visible(False)
    #     ax_box.set_zorder(1000)
    ax_box.patch.set_alpha(0.5)
    ax_box.patch.set_color('white')

    ax_score = fig.add_axes([0.35, 0.65, 0.3, 0.3])
    cmap = plt.get_cmap("tab10")
    sim_freq = 100
    ts = np.arange(len(score_filtered)) / sim_freq
    ax_score.plot(ts, score_filtered)
    ax_score.set_xlabel(r'$t$ (s)')
    ax_score.set_ylim(0, 1)
    ax_score.set_ylabel(r"Time-averaged score")
    ax_score.set_xlim(0, 10)

    epsilon = 0.01
    ax_score.xaxis.set_ticks(np.arange(0, 10 + epsilon, 1))
    ax_score.yaxis.set_ticks(np.arange(0, 1 + epsilon, 0.1))

    score_interp = interpolate.interp1d(ts, score_filtered)

    if t < 10:
        ax_score.axvline(x=t, color=cmap(2))
        inst_score = score_interp(t)
        if inst_score > -1:
            ax_score.plot((0, t), (inst_score, inst_score), color=cmap(2))

    ax_score.patch.set_alpha(0.8)

    fig.savefig(f'{recon_dir}/{input_prefix}-graph-{frame_id}.png', dpi=my_dpi)
    #     [left, bottom, width, height]

    plt.close('all')


plt.rcParams.update({'font.size': 22})

recon_dir = args.recon_directory

metric = args.metric
error_history = np.load(f'{recon_dir}/{metric}_error.npy')
baseline_history = np.load(f'{recon_dir}/{metric}_baseline.npy')
num_samples_history = np.load(f'{recon_dir}/{metric}_num_samples.npy')

baseline_accm = 0
error_accm = 0

for frame_id in range(len(error_history)):
    num_samples = num_samples_history[frame_id]
    if num_samples > 0:
        baseline_accm += baseline_history[frame_id] / num_samples
        error_accm += error_history[frame_id] / num_samples
    # print(num_samples, baseline_accm, error_accm)

print('Score', error_accm / baseline_accm)

for frame_id in range(300):
    render_graph(recon_dir, frame_id, args.input_prefix, args.filter,
                 error_history, baseline_history)
