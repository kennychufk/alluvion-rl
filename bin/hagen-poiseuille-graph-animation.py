import subprocess
import argparse
import os

import numpy as np
from numpy import linalg as LA
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import interpolate
from util import Unit
from util import populate_plt_settings, get_column_width, get_fig_size, get_latex_float

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm

parser = argparse.ArgumentParser(description='Hagen poiseuille plotting')
parser.add_argument('--scene-dir', type=str, required=True)
parser.add_argument('--sequence-label', type=str, required=True)
parser.add_argument('--plot-optimized', type=bool, default=False)
args = parser.parse_args()

matplotlib.use('Agg')

plt.rcParams.update({'font.size': 22})

scene_dir = args.scene_dir
sequence_label = args.sequence_label
plot_optimized = args.plot_optimized

unit = Unit(real_kernel_radius=0.0025,
            real_density0=1000,
            real_gravity=-9.80665)

ground_truth = np.load(f'{scene_dir}/ground_truth.npy')
accelerations = np.load(f'{scene_dir}/accelerations.npy')

rs = np.load(f'{scene_dir}/rs.npy')
ts = np.load(f'{scene_dir}/ts.npy')
full_ts = np.load(f'{scene_dir}/ts-{sequence_label}.npy')
checkpoint_step_ids = np.load(f'{scene_dir}/checkpoint_step_ids.npy')

scale_velocity = 1000
scale_r = 1000
rs_scaled = unit.to_real_length(rs) * scale_r

num_steps = checkpoint_step_ids[-1] + 1

simulated = np.zeros((ground_truth.shape[0], num_steps, ground_truth.shape[2]),
                     ground_truth.dtype)
for step_id in range(num_steps):
    simulated[0][step_id] = np.load(
        f'{scene_dir}/vx-{sequence_label}-{step_id}.npy')

cmap = plt.get_cmap("tab10")

my_dpi = 60


def get_fig_with_background(background_filename, width, height):
    fig = plt.figure(figsize=[width // my_dpi, height // my_dpi],
                     dpi=my_dpi,
                     frameon=False)
    img_ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(img_ax)
    img = plt.imread(background_filename)

    img_ax.imshow(img, cmap='gray')
    img_ax.set_axis_off()
    return fig


for step_id in range(-1, num_steps + 2):
    nominal_step_id = step_id
    if nominal_step_id < 0:
        nominal_step_id = 0
    elif nominal_step_id >= num_steps:
        nominal_step_id = num_steps - 1
    plot_error = (step_id == num_steps + 1)

    fig = get_fig_with_background(
        f'{scene_dir}/particle-{sequence_label}-{nominal_step_id}.png', 1920,
        1080)

    # ax_box = fig.add_axes([0.305, 0.585, 0.355, 0.379])
    # ax_box.xaxis.set_visible(False)
    # ax_box.yaxis.set_visible(False)
    # #     ax_box.set_zorder(1000)
    # ax_box.patch.set_alpha(0.5)
    # ax_box.patch.set_color('white')

    ax = fig.add_axes([0.61, 0.3, 0.3, 0.3])  #left, bottom, width, height
    time_label_pos_list = [(0.5, 0.010), (0.5, 0.0235), (0.5, 0.030)]

    for acc_id, acc in enumerate(accelerations):
        ax.set_title(f'Velocity Profile', y=1.04)

        if not plot_error:
            for t_id, t in enumerate(ts):
                checkpoint_step_id = checkpoint_step_ids[t_id]
                if step_id >= checkpoint_step_id or not plot_optimized:
                    ax.plot(rs_scaled,
                            unit.to_real_velocity(ground_truth[acc_id][t_id]) *
                            scale_velocity,
                            c=cmap(t_id),
                            linewidth=4,
                            alpha=0.3)
                    ax.annotate(r"$t=" + f"{unit.to_real_time(ts[t_id]):.2f}" +
                                r"$s",
                                xy=time_label_pos_list[t_id],
                                xycoords='data',
                                bbox=dict(boxstyle="round,pad=0.2",
                                          facecolor="white",
                                          edgecolor=cmap(t_id)))

            if step_id >= 0 and step_id < num_steps:
                vx_scaled = unit.to_real_velocity(
                    simulated[acc_id][step_id]) * scale_velocity
                ax.plot(rs_scaled, vx_scaled, c='black')

                vx_inerpolate = interpolate.interp1d(rs_scaled, vx_scaled)
                current_annotation_x = 12
                current_annotation_y = vx_inerpolate(
                    current_annotation_x) + 0.002
                ax.annotate(r"$t=" + f"{full_ts[step_id]:.2f}" + r"$s",
                            xy=(current_annotation_x, current_annotation_y),
                            xycoords="data",
                            bbox=dict(boxstyle="round,pad=0.2",
                                      facecolor="white",
                                      edgecolor='black'))

            for t_id, t in enumerate(ts):
                checkpoint_step_id = checkpoint_step_ids[t_id]
                line, = ax.plot(rs_scaled,
                                unit.to_real_velocity(
                                    simulated[acc_id][checkpoint_step_id]) *
                                scale_velocity,
                                c=cmap(t_id),
                                label=f"{unit.to_real_time(t):.2f}s")
                line.set_visible(step_id >= checkpoint_step_id
                                 and not plot_optimized)

        else:
            for t_id, t in enumerate(ts):
                checkpoint_step_id = checkpoint_step_ids[t_id]
                ax.fill_between(
                    rs_scaled,
                    unit.to_real_velocity(ground_truth[acc_id][t_id]) *
                    scale_velocity,
                    unit.to_real_velocity(
                        simulated[acc_id][checkpoint_step_id]) *
                    scale_velocity,
                    color=cmap(t_id))
        ax.set_xlabel(r'$r$ (mm)')
        ax.set_xlim(0)
    ax.set_ylabel(r"$v_{y}$ (mm/s)", rotation=0, labelpad=75)
    ax.set_ylim(0, 0.035)
    line_type_legends = [
        Line2D([0], [0], color=cmap(0), label='Simulated'),
        Line2D([0], [0],
               color=cmap(0),
               linewidth=4,
               alpha=0.3,
               label='Theoretical')
    ]
    if not plot_error and not plot_optimized:
        ax.legend(handles=line_type_legends)

    if step_id < 0:
        fig.savefig(f'{scene_dir}/combined-{sequence_label}-initial.png',
                    dpi=my_dpi)
    elif step_id == num_steps:
        fig.savefig(f'{scene_dir}/combined-{sequence_label}-final.png',
                    dpi=my_dpi)
    elif step_id == num_steps + 1:
        fig.savefig(f'{scene_dir}/combined-{sequence_label}-error.png',
                    dpi=my_dpi)
    else:
        fig.savefig(f'{scene_dir}/combined-{sequence_label}-{step_id}.png',
                    dpi=my_dpi)
    plt.close('all')
