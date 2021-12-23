import os
import sys
import shutil
import glob
import numpy as np
from numpy import linalg as LA
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
containing_dir = f'/media/kennychufk/vol1bk0/{piv_dir_name}/'
pos = np.load(f'{containing_dir}/mat_results/pos.npy')
vel = np.load(f'{containing_dir}/mat_results/vel_original.npy')

image_dim = 1024
calxy = np.load(f'{containing_dir}/calxy.npy').item()
offset_x = np.load(f'{containing_dir}/offset_x.npy').item()
offset_y = np.load(f'{containing_dir}/offset_y.npy').item()
num_points_per_row = pos.shape[1]
pix_point_interval = image_dim / (num_points_per_row + 1)
pos_x = pos[..., 0]
pix_x = (pos[..., 0] + offset_x) / calxy
pix_y = image_dim + ((-pos[..., 1] - offset_y) / calxy)
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
if is_stirrer:
    with open(f"{containing_dir}/robot-time-offset") as f:
        piv_offset = int(f.read())
        print('offset =', piv_offset)


def render_field(containing_dir, frame_prefix, frame_id, pix_x, pix_y, cmap):
    my_dpi = 128
    robot_tid = int((frame_id + piv_offset) / piv_freq * robot_freq)
    if robot_tid >= 4000:
        robot_tid = 3999
    source_img_filename = f"{containing_dir}/{frame_prefix}{frame_id+1:06d}.tif"
    fig = plt.figure(figsize=[image_dim // my_dpi, image_dim // my_dpi],
                     dpi=my_dpi,
                     frameon=False)
    img_ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(img_ax)
    source_img = plt.imread(source_img_filename)

    window_size = 44
    analyze_x0 = int(pix_x[0][0]) - window_size // 2
    analyze_x1 = int(pix_x[0][-1]) + window_size // 2
    analyze_y0 = int(pix_y[0][0]) - window_size // 2
    analyze_y1 = int(pix_y[-1][0]) + window_size // 2

    num_windows_x = (analyze_x1 - analyze_x0) // window_size
    num_windows_y = (analyze_y1 - analyze_y0) // window_size

    analyze_img = np.array(
        source_img[analyze_y0:analyze_y1, analyze_x0:analyze_x1] / 4096,
        dtype=np.float32)
    #     img_ax.imshow(analyze_img, cmap='gray')
    #     img_ax.set_axis_off()

    img_ax.imshow(source_img, cmap='gray')
    img_ax.set_axis_off()
    u = np.copy(vel[frame_id, ..., 0])
    v = np.copy(vel[frame_id, ..., 1])
    acceleration_mask = np.ones(u.shape, dtype=bool)
    if frame_id > 0:
        vel_diff = vel[frame_id] - vel[frame_id - 1]
        acceleration_mask = LA.norm(vel_diff, axis=2) < 0.08

    velocity_mask = LA.norm(vel[frame_id], axis=2) < 0.18
    agitator_mask = np.abs(pos_x - agitator_x[robot_tid]) > 0.012

    brightness_mask = np.ones(u.shape, dtype=bool)
    for window_x in range(num_windows_x):
        for window_y in range(num_windows_y):
            brightness = np.mean(
                analyze_img[window_y * window_size:(window_y + 1) *
                            window_size, window_x *
                            window_size:(window_x + 1) * window_size])
            brightness_mask[window_y, window_x] = (brightness < 0.1)

    mask = agitator_mask & brightness_mask & velocity_mask & acceleration_mask
    mag = np.hypot(u, v)
    q = img_ax.quiver(pix_x[mask],
                      pix_y[mask],
                      u[mask],
                      v[mask],
                      mag[mask],
                      cmap=cmap,
                      scale=5)
    agitator_x_pix = (agitator_x[robot_tid] + offset_x) / calxy
    if 0 <= agitator_x_pix and agitator_x_pix < image_dim:
        img_ax.vlines(agitator_x_pix, 0, image_dim - 1)
    fig.savefig(f'{containing_dir}/gridfield/{frame_id}.png', dpi=my_dpi)
    plt.close('all')
    return mask


os.makedirs(f'{containing_dir}/gridfield', exist_ok=True)
num_frames = len(vel)
masks = Parallel(n_jobs=8)(delayed(render_field)(containing_dir, frame_prefix,
                                                 frame_id, pix_x, pix_y, cmap)
                           for frame_id in range(num_frames))
np.save(f'{containing_dir}/mat_results/mask.npy',
        np.array(masks).astype(np.uint32))
