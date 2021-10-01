import os
import shutil
import glob
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl
import scipy.io as sio
import seaborn as sns
import matplotlib
import matplotlib
from joblib import Parallel, delayed
matplotlib.use('Agg')


def read_row_range(containing_dir):
    typevector = np.array(
        sio.loadmat(f'{containing_dir}/typevector_original.mat')
        ['typevector_original'])
    mask = typevector[0][0]
    start_row_id = -1
    end_row_id = -1
    for row_id, row in enumerate(mask):
        if start_row_id < 0 and (np.sum(row) == len(row)):
            start_row_id = row_id
        elif start_row_id >= 0 and (np.sum(row) < len(row)):
            end_row_id = row_id
            break
    return (start_row_id, end_row_id)


containing_dir = '/media/kennychufk/vol1bk0/20210409_141511'
row_range = read_row_range(containing_dir)
pos = np.load(f'{containing_dir}/pos.npy')
vel = np.load(f'{containing_dir}/vel_filtered.npy')

image_dim = 1024
num_points_per_row = pos.shape[1]
pix_point_interval = image_dim / (num_points_per_row + 1)
pix_x_full, pix_y_full = np.meshgrid(
    np.arange(pix_point_interval, image_dim, pix_point_interval),
    np.arange(pix_point_interval, image_dim, pix_point_interval))
pix_x = pix_x_full[row_range[0]:row_range[1]]
pix_y = pix_y_full[row_range[0]:row_range[1]]

frame_prefix = '20210409_141511'

cmap = sns.color_palette("viridis", as_cmap=True)


def render_field(containing_dir, frame_prefix, frame_id, pix_x, pix_y, cmap):
    my_dpi = 128
    source_img = f"{containing_dir}/{frame_prefix}{frame_id+1:06d}.tif"
    fig = plt.figure(figsize=[image_dim // my_dpi, image_dim // my_dpi],
                     dpi=my_dpi,
                     frameon=False)
    img_ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(img_ax)

    img_ax.imshow(plt.imread(source_img), cmap='gray')
    img_ax.set_axis_off()
    u = vel[frame_id, ..., 0]
    v = vel[frame_id, ..., 1]
    mag = np.hypot(u, v)
    q = img_ax.quiver(pix_x, pix_y, u, v, mag, cmap=cmap)
    # q = img_ax.streamplot(pix_x, pix_y, u, v, cmap=cmap)
    # q = img_ax.streamplot(pix_x, pix_y, u, v, cmap=cmap, lw = 5 * speed / speed.max)
    fig.savefig(
        f'/media/kennychufk/vol1bk0/20210409_141511/gridfield/{frame_id}.png',
        dpi=my_dpi)
    plt.close('all')


num_frames = len(vel)
Parallel(n_jobs=8)(delayed(render_field)(containing_dir, frame_prefix,
                                         frame_id, pix_x, pix_y, cmap)
                   for frame_id in range(num_frames))
