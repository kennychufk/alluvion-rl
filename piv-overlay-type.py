import os
import sys
import shutil
import glob
import numpy as np
from numpy import linalg as LA
from scipy import ndimage
from scipy import interpolate
import subprocess
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl
import scipy.io as sio
import seaborn as sns
import matplotlib
import pandas as pd
from joblib import Parallel, delayed
matplotlib.use('Agg')

piv_dir_name = sys.argv[1]
remove_robot_agitator = True
containing_dir = f'/media/kennychufk/vol1bk0/{piv_dir_name}/'
pos = np.load(f'{containing_dir}/mat_results/pos.npy')
vel = np.load(f'{containing_dir}/mat_results/vel_filtered.npy')

image_dim = 1024
calxy = np.load(f'{containing_dir}/calxy.npy').item()
offset_x = np.load(f'{containing_dir}/offset_x.npy').item()
offset_y = np.load(f'{containing_dir}/offset_y.npy').item()
num_points_per_row = pos.shape[1]
pix_point_interval = image_dim / (num_points_per_row + 1)
pos_x = pos[..., 0]
pix_x = np.rint((pos[..., 0] + offset_x) / calxy).astype(int)
pix_y = image_dim + np.rint((-pos[..., 1] - offset_y) / calxy).astype(int)
if remove_robot_agitator:
    robot_filename = glob.glob(f'{containing_dir}/Trace*.csv')[0]
    robot_frames = pd.read_csv(
        robot_filename,
        comment='"',
        names=['tid', 'j0', 'j1', 'j2', 'j3', 'j4', 'j5', 'x', 'y', 'z', 'v'])
    agitator_x = robot_frames['y'].to_numpy() / 1000.0

hyphen_position = piv_dir_name.find('-')
frame_prefix = piv_dir_name if hyphen_position < 0 else piv_dir_name[:
                                                                     hyphen_position]

cmap = sns.color_palette("vlag", as_cmap=True)

piv_freq = 500.0
robot_freq = 200.0

is_stirrer = True
if is_stirrer and remove_robot_agitator:
    with open(f"{containing_dir}/robot-time-offset") as f:
        piv_offset = int(f.read())
        print('offset =', piv_offset)


def render_field(containing_dir, frame_prefix, frame_id, pix_x, pix_y, cmap,
                 visualize):
    my_dpi = 128
    img_a_filename = f"{containing_dir}/{frame_prefix}{frame_id+1:06d}.tif"
    if visualize:
        fig = plt.figure(figsize=[image_dim // my_dpi, image_dim // my_dpi],
                         dpi=my_dpi,
                         frameon=False)
        img_ax = plt.Axes(fig, [0., 0., 1., 1.])
        fig.add_axes(img_ax)
    img_a = plt.imread(img_a_filename)

    window_size = 44
    analyze_x0 = int(pix_x[0][0]) - window_size // 2
    analyze_x1 = int(pix_x[0][-1]) + window_size // 2
    analyze_y0 = int(pix_y[0][0]) - window_size // 2
    analyze_y1 = int(pix_y[-1][0]) + window_size // 2

    num_windows_x = (analyze_x1 - analyze_x0) // window_size
    num_windows_y = (analyze_y1 - analyze_y0) // window_size

    img_a_roi = np.array(img_a[analyze_y0:analyze_y1, analyze_x0:analyze_x1] /
                         4096,
                         dtype=np.float32)

    if visualize:
        img_ax.imshow(img_a, cmap='gray')
        img_ax.set_axis_off()
    uv = np.copy(vel[frame_id])
    u = np.copy(vel[frame_id, ..., 0])
    v = np.copy(vel[frame_id, ..., 1])

    valid_mask_inv = np.logical_or(np.isnan(u), np.isnan(v))
    valid_mask = np.logical_not(valid_mask_inv)
    # img_ax.scatter(pix_x[valid_mask_inv],
    #               pix_y[valid_mask_inv],
    #               c='#888888',
    #               marker='D')

    # acceleration_mask = np.ones(u.shape, dtype=bool)
    # if frame_id > 0:
    #     vel_diff = vel[frame_id] - vel[frame_id - 1]
    #     acceleration_mask = LA.norm(vel_diff, axis=2) < 0.08
    # acceleration_mask_inv = np.logical_not(acceleration_mask)
    # img_ax.scatter(pix_x[acceleration_mask_inv],
    #               pix_y[acceleration_mask_inv],
    #               c='#991111',
    #               marker='x')

    velocity_mask = LA.norm(vel[frame_id], axis=2) < 0.18
    agitator_mask = np.ones(u.shape, dtype=bool)
    if remove_robot_agitator:
        robot_tid = int((frame_id + piv_offset) / piv_freq * robot_freq)
        if robot_tid >= 4000:
            robot_tid = 3999
        agitator_mask = np.abs(
            pos_x - agitator_x[robot_tid]
        ) > 0.02  # match with agitator_exclusion_dist in cut-silhouette.py

    uv_norm = LA.norm(uv, axis=2)
    uv_norm_std = np.nanstd(uv_norm)
    uv_norm_mean = np.nanmean(uv_norm)
    std_threshold = 1
    velocity_std_mask = uv_norm > (uv_norm_mean - std_threshold * uv_norm_std)

    brightness_mask = np.ones(u.shape, dtype=bool)
    for window_x in range(num_windows_x):
        for window_y in range(num_windows_y):
            brightness = np.mean(
                img_a_roi[window_y * window_size:(window_y + 1) * window_size,
                          window_x * window_size:(window_x + 1) * window_size])
            brightness_mask[window_y, window_x] = (brightness < 0.1)
    brightness_mask_inv = np.logical_not(brightness_mask)
    # img_ax.scatter(pix_x[brightness_mask_inv],
    #               pix_y[brightness_mask_inv],
    #               c='#aa9921',
    #               marker='*')

    # # ==== contrast mask PIVlab
    # img_b_filename = f"{containing_dir}/{frame_prefix}{frame_id+2:06d}.tif"
    # img_b = plt.imread(img_b_filename)
    # img_c = img_a + img_b
    # img_c = img_c / np.max(img_c)
    # gy = np.diff(img_c, 1, 0)
    # gy = np.vstack((gy, gy[-1]))
    # gx = np.diff(img_c, 1, 1)
    # gx =np.hstack((gx, gx[:, -1, np.newaxis]))
    # g=(np.abs(gx)+np.abs(gy))/2
    # filter_size = (np.array([img_a.shape[0]/pix_x.shape[1]+0.5])).astype(int)[0]
    # gb = ndimage.uniform_filter(g, size=filter_size, mode='nearest')
    # gq=np.zeros_like(u)
    # for i in range(pix_x.shape[1]):
    #     for j in range(len(pix_y)):
    #         gq[j, i] = gb[pix_y[j][0], pix_x[0][i]]
    #         # print(j, i, gq[j, i])
    # percentile_10_position = pix_x.shape[0] * pix_x.shape[1] * 0.4
    # index0 = int(percentile_10_position)
    # fraction0 = percentile_10_position - index0
    # percentile_10_partitioned = np.partition(gq.ravel(), (index0, index0+1))
    # percentile_10 = percentile_10_partitioned[index0] * (1-fraction0) + percentile_10_partitioned[index0+1] * fraction0
    # contrast_mask = gq > percentile_10
    # contrast_mask_inv = np.logical_not(contrast_mask)
    # img_ax.scatter(pix_x[contrast_mask_inv],
    #               pix_y[contrast_mask_inv],
    #               c='#10bb05',
    #               marker='x')
    # print(percentile_10)

    mask = agitator_mask & brightness_mask & velocity_mask & valid_mask & velocity_std_mask
    mag = np.hypot(u, v)
    if visualize:
        q = img_ax.quiver(pix_x[mask],
                          pix_y[mask],
                          u[mask],
                          v[mask],
                          mag[mask],
                          cmap=cmap,
                          scale=5)
    if visualize and remove_robot_agitator:
        agitator_x_pix = (agitator_x[robot_tid] + offset_x) / calxy
        if 0 <= agitator_x_pix and agitator_x_pix < image_dim:
            img_ax.vlines(agitator_x_pix, 0, image_dim - 1)
        fig.savefig(f'{containing_dir}/gridfield/{frame_id}.png', dpi=my_dpi)
        plt.close('all')
    return mask


os.makedirs(f'{containing_dir}/gridfield', exist_ok=True)
num_frames = len(vel)
masks = Parallel(n_jobs=8)(delayed(render_field)(
    containing_dir, frame_prefix, frame_id, pix_x, pix_y, cmap, True)
                           for frame_id in range(num_frames))
np.save(f'{containing_dir}/mat_results/mask.npy',
        np.array(masks).astype(np.uint32))
