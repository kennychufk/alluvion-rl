import os
import sys
import argparse
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

parser = argparse.ArgumentParser(description='PIV masking')
parser.add_argument('--piv-dir-name', type=str, required=True)
parser.add_argument('--recon-v-filename', type=str, required=True)
args = parser.parse_args()

piv_dir_name = args.piv_dir_name

containing_dir = f'/media/kennychufk/vol1bk0/{piv_dir_name}/'
pos = np.load(f'{containing_dir}/mat_results/pos.npy')
vel = np.load(args.recon_v_filename)[..., [2, 1]]

image_dim = 1024
calxy = np.load(f'{containing_dir}/calxy.npy').item()
offset_x = np.load(f'{containing_dir}/offset_x.npy').item()
offset_y = np.load(f'{containing_dir}/offset_y.npy').item()
num_points_per_row = pos.shape[1]
pix_point_interval = image_dim / (num_points_per_row + 1)
pos_x = pos[..., 0]
pix_x = np.rint((pos[..., 0] + offset_x) / calxy).astype(int)
pix_y = image_dim + np.rint((-pos[..., 1] - offset_y) / calxy).astype(int)

hyphen_position = piv_dir_name.find('-')
frame_prefix = piv_dir_name if hyphen_position < 0 else piv_dir_name[:
                                                                     hyphen_position]

cmap = sns.color_palette("vlag", as_cmap=True)

piv_freq = 500.0


def render_field(containing_dir, frame_prefix, frame_id_30, pix_x, pix_y, cmap):
    frame_id = int(frame_id_30 /30 * 500 + 0.5)
    my_dpi = 128
    fig = plt.figure(figsize=[image_dim // my_dpi, image_dim // my_dpi],
                     dpi=my_dpi,
                     frameon=False)
    img_ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(img_ax)
    img_a = np.zeros((image_dim, image_dim))

    img_ax.imshow(img_a, cmap='gray')
    img_ax.set_axis_off()

    u = np.copy(vel[frame_id, ..., 0])
    v = np.copy(vel[frame_id, ..., 1])
    mag = np.hypot(u, v)
    q = img_ax.quiver(pix_x,
                      pix_y,
                      u,
                      v,
                      mag,
                      cmap=cmap,
                      scale=5)
    fig.savefig(
        f'{containing_dir}/gridfield/frame30fps-recon-{frame_id_30}.png',
        dpi=my_dpi)
    plt.close('all')


os.makedirs(f'{containing_dir}/gridfield', exist_ok=True)

Parallel(n_jobs=8)(
    delayed(render_field)(containing_dir, frame_prefix, frame_id_30, pix_x, pix_y, cmap)
    for frame_id_30 in range(30*20))
