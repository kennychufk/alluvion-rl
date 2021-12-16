import os
import sys
import shutil
import glob
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl
import scipy.io as sio
import seaborn as sns
import matplotlib
from joblib import Parallel, delayed
matplotlib.use('Agg')


def render_policy(containing_dir, frame_prefix, action_all, frame_id,
                  num_frames_100):
    max_xoffset = 0.05
    max_focal_dist = 0.2
    max_strength = 4000
    my_dpi = 128
    frame_id_100 = int(frame_id / 30 * 100)
    source_img_filename = f"{containing_dir}/{frame_prefix}{frame_id}.png"
    source_img = plt.imread(source_img_filename)
    image_height, image_width = source_img.shape[:2]

    fig = plt.figure(figsize=[image_width // my_dpi, image_height // my_dpi],
                     dpi=my_dpi,
                     frameon=False)
    img_ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(img_ax)
    img_ax.imshow(source_img)
    img_ax.set_axis_off()

    cmap = plt.get_cmap("tab10")
    focal_point_ax = fig.add_axes([0.5, 0.5, 0.4, 0.4], projection='3d')
    focal_point_ax.patch.set_alpha(0.5)
    focal_point_ax.set_xlim(-max_xoffset, max_xoffset)
    focal_point_ax.set_ylim(-max_xoffset, max_xoffset)
    focal_point_ax.set_zlim(-max_xoffset, max_xoffset)
    action = action_all[frame_id_100]
    points = action[:, :3]
    #     buoy_focal_
    for buoy_id in range(len(action)):
        points = np.zeros((4, 3))
        points[0] = action[buoy_id, 0:3]
        points[1] = action[buoy_id, 6:9]
        points[2] = action[buoy_id, 12:15]
        points[3] = points[0]
        focal_point_ax.plot(points[:, 0],
                            points[:, 1],
                            points[:, 2],
                            color=cmap(buoy_id),
                            label=f'{buoy_id}')

    strength_ax = fig.add_axes([0.1, 0.5, 0.4, 0.4])
    strength_ax.patch.set_alpha(0.5)

    half_window_frame = 50
    half_window_time = 50 / 100
    plot_range_start = np.max([frame_id_100 - half_window_frame, 0])
    plot_range_end = np.min(
        [frame_id_100 + half_window_frame + 1, num_frames_100])
    t = (np.arange(plot_range_start, plot_range_end)) / 100
    strength_ax.set_ylim(0, max_strength)
    strength_ax.set_xlim(plot_range_start / 100, plot_range_end / 100)
    for buoy_id in range(len(action)):
        strength_ax.plot(
            t, action_all[plot_range_start:plot_range_end, buoy_id, 20])

    fig.savefig(f'{containing_dir}/overlay{frame_id}.png', dpi=my_dpi)
    plt.close('all')


render_image_dir = '/home/kennychufk/workspace/pythonWs/test-run-al-outside/truth/rltruth-27e78d56-1026.13.28.06'
render_image_prefix = 'renreconfieldmasked'
val_dir = '/home/kennychufk/workspace/pythonWs/alluvion-optim/val'

num_frames_100 = 999
action_dim = np.load(f'{val_dir}/act-0.npy').shape
action_all = np.zeros((num_frames_100, *action_dim))
for i in range(num_frames_100):
    action_all[i] = np.load(f'{val_dir}/act-{i}.npy')

num_frames = 300
Parallel(n_jobs=8)(
    delayed(render_policy)(render_image_dir, render_image_prefix, action_all,
                           frame_id, num_frames_100)
    # for frame_id in range(186, 187))
    for frame_id in range(num_frames))
