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

image_dim = 1024
calxy = np.load(f'{containing_dir}/calxy.npy').item()
offset_x = np.load(f'{containing_dir}/offset_x.npy').item()
offset_y = np.load(f'{containing_dir}/offset_y.npy').item()
robot_filename = glob.glob(f'{containing_dir}/Trace*.csv')[0]
robot_frames = pd.read_csv(
    robot_filename,
    comment='"',
    names=['tid', 'j0', 'j1', 'j2', 'j3', 'j4', 'j5', 'x', 'y', 'z', 'v'])
agitator_x = robot_frames['y'].to_numpy() / 1000.0

frame_prefix = piv_dir_name

cmap = sns.color_palette("vlag", as_cmap=True)

piv_freq = 500.0
robot_freq = 200.0

blank_img = np.ones((image_dim, image_dim))


def render_robot(containing_dir, frame_prefix, frame_id, cmap):
    my_dpi = 128
    robot_tid = int(frame_id / piv_freq * robot_freq)
    if robot_tid >= 4000:
        return
    fig = plt.figure(figsize=[image_dim // my_dpi, image_dim // my_dpi],
                     dpi=my_dpi,
                     frameon=False)
    img_ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(img_ax)

    img_ax.imshow(blank_img, cmap='gray')
    img_ax.set_axis_off()
    agitator_x_pix = (agitator_x[robot_tid] + offset_x) / calxy
    if 0 <= agitator_x_pix and agitator_x_pix < image_dim:
        img_ax.vlines(agitator_x_pix, 0, image_dim - 1)
    fig.savefig(f'{containing_dir}/gridfield/robot{frame_id}.png', dpi=my_dpi)
    plt.close('all')


os.makedirs(f'{containing_dir}/gridfield', exist_ok=True)
Parallel(n_jobs=8)(
    delayed(render_robot)(containing_dir, frame_prefix, frame_id, cmap)
    for frame_id in range(10000))
