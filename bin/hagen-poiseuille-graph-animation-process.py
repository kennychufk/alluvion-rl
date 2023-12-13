import subprocess
import argparse
import os
import glob

import numpy as np
from numpy import linalg as LA
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import interpolate
from util import Unit
from util import populate_plt_settings, get_column_width, get_fig_size, get_latex_float
import wandb

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib import patches

matplotlib.use('Agg')

plt.rcParams.update({'font.size': 22})

scene_dir = '/home/kennychufk/workspace/pythonWs/vis-opt-particles-final'
attr_dir = '/home/kennychufk/workspace/pythonWs/vis-opt-particles'

unit = Unit(real_kernel_radius=0.0025,
            real_density0=1000,
            real_gravity=-9.80665)

ground_truth = np.load(f'{attr_dir}/ground_truth.npy')
accelerations = np.load(f'{attr_dir}/accelerations.npy')

rs = np.load(f'{attr_dir}/rs.npy')
ts = np.load(f'{attr_dir}/ts.npy')
checkpoint_step_ids = np.load(f'{attr_dir}/checkpoint_step_ids.npy')

scale_velocity = 1000
scale_r = 1000
rs_scaled = unit.to_real_length(rs) * scale_r

num_checkpoints = len(checkpoint_step_ids)
num_optimization_steps = 800
simulated = np.zeros((num_optimization_steps, ground_truth.shape[0],
                      num_checkpoints, ground_truth.shape[2]),
                     ground_truth.dtype)
for optimization_step_id in range(num_optimization_steps):
    for checkpoint_id in range(num_checkpoints):
        step_id = checkpoint_step_ids[checkpoint_id]
        vx_filename = glob.glob(
            f'{scene_dir}/vx-{optimization_step_id}-*-False-{step_id}.npy')[0]
        simulated[optimization_step_id,
                  0][checkpoint_id] = np.load(vx_filename)

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


api = wandb.Api()
run = api.run('kennychufk/alluvion/1egs88bw')
history = run.scan_history()

vis = np.zeros(num_optimization_steps)
bvis = np.zeros(num_optimization_steps)
vis_grad = np.zeros(num_optimization_steps)
bvis_grad = np.zeros(num_optimization_steps)
loss = np.zeros(num_optimization_steps)

for row in history:
    step = row['_step']
    if step >= num_optimization_steps:
        continue
    if 'vis_real' in row:
        vis[step] = row['vis_real']
    if 'bvis_real' in row:
        bvis[step] = row['bvis_real']
    if '∇vis' in row:
        vis_grad[step] = row['∇vis']
    if '∇bvis' in row:
        bvis_grad[step] = row['∇bvis']
    if 'loss' in row:
        loss[step] = row['loss']

for optimization_step_id in range(-1, num_optimization_steps):
    nominal_step_id = optimization_step_id
    plot_initial = optimization_step_id < 0
    if plot_initial:
        nominal_step_id = 0

    fig = get_fig_with_background(
        f'{scene_dir}/opt-process-{nominal_step_id}.png', 1920, 1080)

    ax = fig.add_axes([0.69, 0.3, 0.3, 0.3])  #left, bottom, width, height
    time_label_pos_list = [(0.5, 0.010), (0.5, 0.0235), (0.5, 0.030)]
    for acc_id, acc in enumerate(accelerations):
        ax.set_title(f'Velocity Profile', y=1.04)
        for t_id, t in enumerate(ts):
            ax.plot(rs_scaled,
                    unit.to_real_velocity(ground_truth[acc_id][t_id]) *
                    scale_velocity,
                    c=cmap(t_id),
                    linewidth=4,
                    alpha=0.3)
            ax.annotate(r"$t=" + f"{unit.to_real_time(ts[t_id]):.2f}" + r"$s",
                        xy=time_label_pos_list[t_id],
                        xycoords='data',
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor="white",
                                  edgecolor=cmap(t_id)))
            line, = ax.plot(rs_scaled,
                            unit.to_real_velocity(simulated[nominal_step_id,
                                                            acc_id][t_id]) *
                            scale_velocity,
                            c=cmap(t_id),
                            label=f"{unit.to_real_time(t):.2f}s")
        ax.set_xlabel(r'$r$ (mm)')
        ax.set_xlim(0)
    ax.set_ylabel(r"$v_{y}$ (mm/s)", rotation='horizontal')
    ax.yaxis.set_label_coords(0, 1.06)
    ax.set_ylim(0, 0.035)
    line_type_legends = [
        Line2D([0], [0], color=cmap(0), label='Simulated'),
        Line2D([0], [0],
               color=cmap(0),
               linewidth=4,
               alpha=0.3,
               label='Theoretical')
    ]
    ax.legend(handles=line_type_legends)

    scale_vis = 1e6
    vis_color = 'tab:blue'
    bvis_color = 'tab:red'

    vis_ax = fig.add_axes([0.03, 0.55, 0.3, 0.3])  #left, bottom, width, height
    vis_ax.set_ylabel(r"$\nu_{\mathcal{F}}$ (cSt)",
                      rotation='horizontal',
                      color=vis_color)
    vis_ax.yaxis.set_label_coords(0, 1.06)
    vis_ax.tick_params(axis='y', labelcolor=vis_color)
    vis_ax.set_xlim(0, num_optimization_steps)
    vis_ax.set_ylim(2, 2.5)

    tick_label_offset = matplotlib.transforms.ScaledTranslation(
        0, -0.1, fig.dpi_scale_trans)
    # apply offset transform to all x ticklabels.
    for label in vis_ax.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + tick_label_offset)

    bvis_ax = vis_ax.twinx()
    bvis_ax.set_ylabel(r"$\nu_{\mathcal{B}}$ (cSt)",
                       rotation='horizontal',
                       color=bvis_color)
    bvis_ax.yaxis.set_label_coords(1, 1.11)

    bvis_ax.tick_params(axis='y', labelcolor=bvis_color)
    bvis_ax.set_xlim(0, num_optimization_steps)
    bvis_ax.set_ylim(3.2, 4.2)

    vis_grad_ax = fig.add_axes([0.03, 0.13, 0.3,
                                0.3])  #left, bottom, width, height
    vis_grad_ax.set_ylabel(
        r"$\frac{\partial \mathrm{MSE}}{\partial \nu_{\mathcal{F}}} (\mathrm{s}^{-1})$",
        rotation='horizontal',
        color=vis_color,
        fontsize=32)
    vis_grad_ax.yaxis.set_label_coords(0.05, 1.06)
    vis_grad_ax.tick_params(axis='y', labelcolor=vis_color)
    vis_grad_ax.set_ylim(-3e-6, 6e-6)
    vis_grad_ax.set_xlim(0, num_optimization_steps)
    vis_grad_ax.yaxis.set_major_formatter(
        matplotlib.ticker.ScalarFormatter(useMathText=True))
    vis_grad_ax.set_xlabel('Optimization steps')

    bvis_grad_ax = vis_grad_ax.twinx()
    bvis_grad_ax.set_ylabel(
        r"$\frac{\partial \mathrm{MSE}}{\partial \nu_{\mathcal{B}}} (\mathrm{s}^{-1})$",
        rotation='horizontal',
        color=bvis_color,
        fontsize=32)
    bvis_grad_ax.yaxis.set_label_coords(0.95, 1.20)
    bvis_grad_ax.tick_params(axis='y', labelcolor=bvis_color)
    bvis_grad_ax.set_ylim(-3.5e-7, 7e-7)
    bvis_grad_ax.set_xlim(0, num_optimization_steps)
    bvis_grad_ax.yaxis.set_major_formatter(
        matplotlib.ticker.ScalarFormatter(useMathText=True))
    if not plot_initial:
        bvis_ax.axvline(x=nominal_step_id, color='black')
        bvis_grad_ax.axvline(x=nominal_step_id, color='black')
        vis_ax.plot(vis[:nominal_step_id + 1] * scale_vis, color=vis_color)
        bvis_ax.plot(bvis[:nominal_step_id + 1] * scale_vis, color=bvis_color)
        vis_grad_ax.plot(vis_grad[:nominal_step_id + 1], color=vis_color)
        bvis_grad_ax.plot(bvis_grad[:nominal_step_id + 1], color=bvis_color)

    bvis_ax.add_patch(
        patches.Rectangle((300, 3.53),
                          200,
                          0.23,
                          lw=1,
                          ec='black',
                          fc='w',
                          zorder=2))
    bvis_ax.annotate(f"Step {nominal_step_id}",
                     xy=(0.5, 0.6),
                     xycoords="axes fraction",
                     va="center",
                     ha="center",
                     c='black')
    bvis_ax.annotate(r"$\nu_{\mathcal{F}}=" +
                     ("{:.3f}".format(vis[nominal_step_id] * scale_vis)) +
                     r"$ cSt",
                     xy=(0.5, 0.5),
                     xycoords="axes fraction",
                     va="center",
                     ha="center",
                     c=vis_color)
    bvis_ax.annotate(r"$\nu_{\mathcal{B}}=" +
                     ("{:.3f}".format(bvis[nominal_step_id] * scale_vis)) +
                     r"$ cSt",
                     xy=(0.5, 0.4),
                     xycoords="axes fraction",
                     va="center",
                     ha="center",
                     c=bvis_color)

    bvis_grad_ax.add_patch(
        patches.Rectangle((220, 0.78e-7),
                          360,
                          3e-7,
                          lw=1,
                          ec='black',
                          fc='w',
                          zorder=2))
    vis_grad_str = ("{:+.2e}".format(vis_grad[nominal_step_id])).replace(
        "e-0", "e-")
    vis_grad_str = vis_grad_str[:5] + r"\times 10^{" + vis_grad_str[6:].lstrip(
        "0") + "}\mathrm{s}^{-1}"
    bvis_grad_str = ("{:+.2e}".format(bvis_grad[nominal_step_id])).replace(
        "e-0", "e-")
    bvis_grad_str = bvis_grad_str[:5] + r"\times 10^{" + bvis_grad_str[
        6:].lstrip("0") + "}\mathrm{s}^{-1}"
    bvis_grad_ax.annotate(
        r"$\frac{\partial \mathrm{MSE}}{\partial \nu_{\mathcal{F}}}=" +
        vis_grad_str + r"$",
        xy=(0.5, 0.6),
        xycoords="axes fraction",
        va="center",
        ha="center",
        c=vis_color)
    bvis_grad_ax.annotate(
        r"$\frac{\partial \mathrm{MSE}}{\partial \nu_{\mathcal{B}}}=" +
        bvis_grad_str + r"$",
        xy=(0.5, 0.5),
        xycoords="axes fraction",
        va="center",
        ha="center",
        c=bvis_color)

    vis_ax.scatter(x=nominal_step_id,
                   y=vis[nominal_step_id] * scale_vis,
                   c=vis_color,
                   s=200)
    bvis_ax.scatter(x=nominal_step_id,
                    y=bvis[nominal_step_id] * scale_vis,
                    c=bvis_color,
                    s=200)
    vis_grad_ax.scatter(x=nominal_step_id,
                        y=vis_grad[nominal_step_id],
                        c=vis_color,
                        s=200)
    bvis_grad_ax.scatter(x=nominal_step_id,
                         y=bvis_grad[nominal_step_id],
                         c=bvis_color,
                         s=200)

    if optimization_step_id < 0:
        fig.savefig(f'{scene_dir}/combined-initial.png', dpi=my_dpi)
    elif optimization_step_id == num_optimization_steps:
        fig.savefig(f'{scene_dir}/combined-final.png', dpi=my_dpi)
    else:
        fig.savefig(f'{scene_dir}/combined-{optimization_step_id}.png',
                    dpi=my_dpi)
    plt.close('all')
