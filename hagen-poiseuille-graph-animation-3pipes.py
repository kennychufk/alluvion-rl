import subprocess
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

matplotlib.use('Agg')

plt.rcParams.update({'font.size': 22})

scene_dir = '/home/kennychufk/workspace/pythonWs/vis-opt-particles'
middle_sequence_label = "0.006386599135621174-0.008941238789869644-False"
left_vis_sequence_label = "0.006450465126977386-0.008941238789869644-True"
right_bvis_sequence_label = "0.006386599135621174-0.00903065117776834-True"

unit = Unit(real_kernel_radius=0.0025,
            real_density0=1000,
            real_gravity=-9.80665)

ground_truth = np.load(f'{scene_dir}/ground_truth.npy')
accelerations = np.load(f'{scene_dir}/accelerations.npy')

rs = np.load(f'{scene_dir}/rs.npy')
ts = np.load(f'{scene_dir}/ts.npy')
full_ts = np.load(f'{scene_dir}/ts-{middle_sequence_label}.npy')
checkpoint_step_ids = np.load(f'{scene_dir}/checkpoint_step_ids.npy')

scale_velocity = 1000
scale_r = 1000
rs_scaled = unit.to_real_length(rs) * scale_r

num_steps = checkpoint_step_ids[-1] + 1

right_bvis_simulated = np.zeros(
    (ground_truth.shape[0], num_steps, ground_truth.shape[2]),
    ground_truth.dtype)
for step_id in range(num_steps):
    right_bvis_simulated[0][step_id] = np.load(
        f'{scene_dir}/vx-{right_bvis_sequence_label}-{step_id}.npy')

left_vis_simulated = np.zeros(
    (ground_truth.shape[0], num_steps, ground_truth.shape[2]),
    ground_truth.dtype)
for step_id in range(num_steps):
    left_vis_simulated[0][step_id] = np.load(
        f'{scene_dir}/vx-{left_vis_sequence_label}-{step_id}.npy')

middle_simulated = np.zeros(
    (ground_truth.shape[0], num_steps, ground_truth.shape[2]),
    ground_truth.dtype)
for step_id in range(num_steps):
    middle_simulated[0][step_id] = np.load(
        f'{scene_dir}/vx-{middle_sequence_label}-{step_id}.npy')

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
        f'{scene_dir}/particle-3pipes-{nominal_step_id}.png', 1920, 1080)

    box_extension = 0.05
    right_ax_box = fig.add_axes([
        0.73 - box_extension, 0.10 - box_extension, 0.25 + box_extension * 1.2,
        0.3 + box_extension * 1.5
    ])
    right_ax_box.xaxis.set_visible(False)
    right_ax_box.yaxis.set_visible(False)
    right_ax_box.patch.set_alpha(0.75)
    right_ax_box.patch.set_color('white')
    right_ax_box.spines['top'].set_visible(False)
    right_ax_box.spines['right'].set_visible(False)
    right_ax_box.spines['bottom'].set_visible(False)
    right_ax_box.spines['left'].set_visible(False)
    right_bvis_ax = fig.add_axes([0.73, 0.10, 0.25,
                                  0.3])  #left, bottom, width, height

    left_ax_box = fig.add_axes([
        0.062 - box_extension, 0.10 - box_extension,
        0.25 + box_extension * 1.2, 0.3 + box_extension * 1.5
    ])
    left_ax_box.xaxis.set_visible(False)
    left_ax_box.yaxis.set_visible(False)
    left_ax_box.patch.set_alpha(0.75)
    left_ax_box.patch.set_color('white')
    left_ax_box.spines['top'].set_visible(False)
    left_ax_box.spines['right'].set_visible(False)
    left_ax_box.spines['bottom'].set_visible(False)
    left_ax_box.spines['left'].set_visible(False)
    left_vis_ax = fig.add_axes([0.062, 0.10, 0.25,
                                0.3])  #left, bottom, width, height

    if plot_error:
        middle_ax_box = fig.add_axes([
            0.395 - box_extension, 0.10 - box_extension,
            0.25 + box_extension * 1.2, 0.3 + box_extension * 1.5
        ])
        middle_ax_box.xaxis.set_visible(False)
        middle_ax_box.yaxis.set_visible(False)
        middle_ax_box.patch.set_alpha(0.75)
        middle_ax_box.patch.set_color('white')
        middle_ax_box.spines['top'].set_visible(False)
        middle_ax_box.spines['right'].set_visible(False)
        middle_ax_box.spines['bottom'].set_visible(False)
        middle_ax_box.spines['left'].set_visible(False)
        middle_ax = fig.add_axes([0.395, 0.10, 0.25,
                                  0.3])  #left, bottom, width, height
    else:
        middle_ax = None

    for acc_id, acc in enumerate(accelerations):
        right_bvis_ax.set_title(f'Velocity Profile')
        left_vis_ax.set_title(f'Velocity Profile')
        if middle_ax is not None:
            middle_ax.set_title(f'Velocity Profile')

        if not plot_error:
            for t_id, t in enumerate(ts):
                right_bvis_ax.plot(
                    rs_scaled,
                    unit.to_real_velocity(ground_truth[acc_id][t_id]) *
                    scale_velocity,
                    c=cmap(t_id),
                    linewidth=4,
                    alpha=0.3)
                left_vis_ax.plot(
                    rs_scaled,
                    unit.to_real_velocity(ground_truth[acc_id][t_id]) *
                    scale_velocity,
                    c=cmap(t_id),
                    linewidth=4,
                    alpha=0.3)
                if middle_ax is not None:
                    middle_ax.plot(
                        rs_scaled,
                        unit.to_real_velocity(ground_truth[acc_id][t_id]) *
                        scale_velocity,
                        c=cmap(t_id),
                        linewidth=4,
                        alpha=0.3)
            right_bvis_ax.annotate(r"$t=" + f"{unit.to_real_time(ts[2]):.2f}" +
                                   r"$s",
                                   xy=(0.5, 0.030),
                                   xycoords='data',
                                   bbox=dict(boxstyle="round,pad=0.2",
                                             facecolor="white",
                                             edgecolor=cmap(2)))
            right_bvis_ax.annotate(r"$t=" + f"{unit.to_real_time(ts[1]):.2f}" +
                                   r"$s",
                                   xy=(0.5, 0.0235),
                                   xycoords='data',
                                   bbox=dict(boxstyle="round,pad=0.2",
                                             facecolor="white",
                                             edgecolor=cmap(1)))
            right_bvis_ax.annotate(r"$t=" + f"{unit.to_real_time(ts[0]):.2f}" +
                                   r"$s",
                                   xy=(0.5, 0.010),
                                   xycoords='data',
                                   bbox=dict(boxstyle="round,pad=0.2",
                                             facecolor="white",
                                             edgecolor=cmap(0)))
            left_vis_ax.annotate(r"$t=" + f"{unit.to_real_time(ts[2]):.2f}" +
                                 r"$s",
                                 xy=(0.5, 0.030),
                                 xycoords='data',
                                 bbox=dict(boxstyle="round,pad=0.2",
                                           facecolor="white",
                                           edgecolor=cmap(2)))
            left_vis_ax.annotate(r"$t=" + f"{unit.to_real_time(ts[1]):.2f}" +
                                 r"$s",
                                 xy=(0.5, 0.0235),
                                 xycoords='data',
                                 bbox=dict(boxstyle="round,pad=0.2",
                                           facecolor="white",
                                           edgecolor=cmap(1)))
            left_vis_ax.annotate(r"$t=" + f"{unit.to_real_time(ts[0]):.2f}" +
                                 r"$s",
                                 xy=(0.5, 0.010),
                                 xycoords='data',
                                 bbox=dict(boxstyle="round,pad=0.2",
                                           facecolor="white",
                                           edgecolor=cmap(0)))
            if middle_ax is not None:
                middle_ax.annotate(r"$t=" + f"{unit.to_real_time(ts[2]):.2f}" +
                                   r"$s",
                                   xy=(0.5, 0.030),
                                   xycoords='data',
                                   bbox=dict(boxstyle="round,pad=0.2",
                                             facecolor="white",
                                             edgecolor=cmap(2)))
                middle_ax.annotate(r"$t=" + f"{unit.to_real_time(ts[1]):.2f}" +
                                   r"$s",
                                   xy=(0.5, 0.0235),
                                   xycoords='data',
                                   bbox=dict(boxstyle="round,pad=0.2",
                                             facecolor="white",
                                             edgecolor=cmap(1)))
                middle_ax.annotate(r"$t=" + f"{unit.to_real_time(ts[0]):.2f}" +
                                   r"$s",
                                   xy=(0.5, 0.010),
                                   xycoords='data',
                                   bbox=dict(boxstyle="round,pad=0.2",
                                             facecolor="white",
                                             edgecolor=cmap(0)))

            if step_id >= 0 and step_id < num_steps:
                right_vx_scaled = unit.to_real_velocity(
                    right_bvis_simulated[acc_id][step_id]) * scale_velocity
                right_bvis_ax.plot(rs_scaled, right_vx_scaled, c='black')
                left_vx_scaled = unit.to_real_velocity(
                    left_vis_simulated[acc_id][step_id]) * scale_velocity
                left_vis_ax.plot(rs_scaled, left_vx_scaled, c='black')
                if middle_ax is not None:
                    middle_vx_scaled = unit.to_real_velocity(
                        middle_simulated[acc_id][step_id]) * scale_velocity
                    middle_ax.plot(rs_scaled, middle_vx_scaled, c='black')

                right_vx_inerpolate = interpolate.interp1d(
                    rs_scaled, right_vx_scaled)
                left_vx_inerpolate = interpolate.interp1d(
                    rs_scaled, left_vx_scaled)
                if middle_ax is not None:
                    middle_vx_inerpolate = interpolate.interp1d(
                        rs_scaled, middle_vx_scaled)
                current_annotation_x = 12
                right_current_annotation_y = right_vx_inerpolate(
                    current_annotation_x) + 0.002
                left_current_annotation_y = left_vx_inerpolate(
                    current_annotation_x) + 0.002
                if middle_ax is not None:
                    middle_current_annotation_y = middle_vx_inerpolate(
                        current_annotation_x) + 0.002
                right_bvis_ax.annotate(
                    r"$t=" + f"{full_ts[step_id]:.2f}" + r"$s",
                    xy=(current_annotation_x, right_current_annotation_y),
                    xycoords="data",
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor="white",
                              edgecolor='black'))
                left_vis_ax.annotate(
                    r"$t=" + f"{full_ts[step_id]:.2f}" + r"$s",
                    xy=(current_annotation_x, left_current_annotation_y),
                    xycoords="data",
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor="white",
                              edgecolor='black'))
                if middle_ax is not None:
                    middle_ax.annotate(
                        r"$t=" + f"{full_ts[step_id]:.2f}" + r"$s",
                        xy=(current_annotation_x, middle_current_annotation_y),
                        xycoords="data",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor="white",
                                  edgecolor='black'))

            for t_id, t in enumerate(ts):
                checkpoint_step_id = checkpoint_step_ids[t_id]
                line, = right_bvis_ax.plot(
                    rs_scaled,
                    unit.to_real_velocity(
                        right_bvis_simulated[acc_id][checkpoint_step_id]) *
                    scale_velocity,
                    c=cmap(t_id),
                    label=f"{unit.to_real_time(t):.2f}s")
                line.set_visible(step_id >= checkpoint_step_id)
                line, = left_vis_ax.plot(
                    rs_scaled,
                    unit.to_real_velocity(
                        left_vis_simulated[acc_id][checkpoint_step_id]) *
                    scale_velocity,
                    c=cmap(t_id),
                    label=f"{unit.to_real_time(t):.2f}s")
                line.set_visible(step_id >= checkpoint_step_id)
                if middle_ax is not None:
                    line, = middle_ax.plot(
                        rs_scaled,
                        unit.to_real_velocity(
                            middle_simulated[acc_id][checkpoint_step_id]) *
                        scale_velocity,
                        c=cmap(t_id),
                        label=f"{unit.to_real_time(t):.2f}s")
                    line.set_visible(step_id >= checkpoint_step_id)

        else:
            for t_id, t in enumerate(ts):
                checkpoint_step_id = checkpoint_step_ids[t_id]
                right_bvis_ax.fill_between(
                    rs_scaled,
                    unit.to_real_velocity(ground_truth[acc_id][t_id]) *
                    scale_velocity,
                    unit.to_real_velocity(
                        right_bvis_simulated[acc_id][checkpoint_step_id]) *
                    scale_velocity,
                    color=cmap(t_id))
                left_vis_ax.fill_between(
                    rs_scaled,
                    unit.to_real_velocity(ground_truth[acc_id][t_id]) *
                    scale_velocity,
                    unit.to_real_velocity(
                        left_vis_simulated[acc_id][checkpoint_step_id]) *
                    scale_velocity,
                    color=cmap(t_id))
                middle_ax.fill_between(
                    rs_scaled,
                    unit.to_real_velocity(ground_truth[acc_id][t_id]) *
                    scale_velocity,
                    unit.to_real_velocity(
                        middle_simulated[acc_id][checkpoint_step_id]) *
                    scale_velocity,
                    color=cmap(t_id))
        right_bvis_ax.set_xlabel(r'$r$ (mm)')
        right_bvis_ax.set_xlim(0)
        left_vis_ax.set_xlabel(r'$r$ (mm)')
        left_vis_ax.set_xlim(0)
        if middle_ax is not None:
            middle_ax.set_xlabel(r'$r$ (mm)')
            middle_ax.set_xlim(0)
    right_bvis_ax.set_ylabel(r"$v_{y}$ (mm/s)")
    right_bvis_ax.set_ylim(0, 0.035)
    left_vis_ax.set_ylabel(r"$v_{y}$ (mm/s)")
    left_vis_ax.set_ylim(0, 0.035)
    if middle_ax is not None:
        middle_ax.set_ylabel(r"$v_{y}$ (mm/s)")
        middle_ax.set_ylim(0, 0.035)
    line_type_legends = [
        Line2D([0], [0], color=cmap(0), label='Simulated'),
        Line2D([0], [0],
               color=cmap(0),
               linewidth=4,
               alpha=0.3,
               label='Theoretical')
    ]
    if not plot_error:
        right_bvis_ax.legend(handles=line_type_legends)
        left_vis_ax.legend(handles=line_type_legends)
        if middle_ax is not None:
            middle_ax.legend(handles=line_type_legends)

    if step_id < 0:
        fig.savefig(f'{scene_dir}/combined-3pipes-initial.png', dpi=my_dpi)
    elif step_id == num_steps:
        fig.savefig(f'{scene_dir}/combined-3pipes-final.png', dpi=my_dpi)
    elif step_id == num_steps + 1:
        fig.savefig(f'{scene_dir}/combined-3pipes-error.png', dpi=my_dpi)
    else:
        fig.savefig(f'{scene_dir}/combined-3pipes-{step_id}.png', dpi=my_dpi)
    plt.close('all')
