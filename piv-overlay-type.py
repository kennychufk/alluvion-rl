import os
import sys
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


def read_typevector_filtered(containing_dir):
    return np.array(
        sio.loadmat(f'{containing_dir}/mat_results/typevector_filtered.mat')
        ['typevector_filtered'])


piv_dir_name = sys.argv[1]
containing_dir = f'/media/kennychufk/vol1bk0/{piv_dir_name}/'
pos = np.load(f'{containing_dir}/mat_results/pos.npy')
vel = np.load(f'{containing_dir}/mat_results/vel_filtered.npy')
typevector_filtered = read_typevector_filtered(containing_dir)

image_dim = 1024
calxy = np.load(f'{containing_dir}/calxy.npy').item()
offset_x = np.load(f'{containing_dir}/offset_x.npy').item()
offset_y = np.load(f'{containing_dir}/offset_y.npy').item()
num_points_per_row = pos.shape[1]
pix_point_interval = image_dim / (num_points_per_row + 1)
pix_x = (pos[..., 0] + offset_x) / calxy
pix_y = image_dim + ((-pos[..., 1] - offset_y) / calxy)

frame_prefix = piv_dir_name

cmap = sns.color_palette("vlag", as_cmap=True)


def render_field(containing_dir, frame_prefix, frame_id, pix_x, pix_y,
                 typevector_filtered, cmap):
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
    is_filtered = typevector_filtered[frame_id][0] - 1
    q = img_ax.quiver(pix_x, pix_y, u, v, is_filtered, cmap=cmap, scale=5)
    fig.savefig(f'{containing_dir}/gridfield/{frame_id}.png', dpi=my_dpi)
    plt.close('all')


os.mkdir(f'{containing_dir}/gridfield')
num_frames = len(vel)
Parallel(n_jobs=8)(
    delayed(render_field)(containing_dir, frame_prefix, frame_id, pix_x, pix_y,
                          typevector_filtered, cmap)
    for frame_id in range(num_frames))
