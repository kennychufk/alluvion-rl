import numpy as np
import scipy.io as sio
import sys


def uv_mat_to_np(u_mat, v_mat):
    uv_np = np.zeros(
        [len(u_mat), len(u_mat[0][0]),
         len(u_mat[0][0][0]), 2], np.float64)
    for time_id in range(len(u_mat)):
        uv_np[time_id] = uv = np.dstack([
            u_mat[time_id][0].astype(np.float64),
            v_mat[time_id][0].astype(np.float64)
        ])
    return uv_np


def read_convert_uv_mat(containing_dir, tag):
    u_mat = np.array(sio.loadmat(f'{containing_dir}/u_{tag}.mat')[f'u_{tag}'])
    v_mat = np.array(sio.loadmat(f'{containing_dir}/v_{tag}.mat')[f'v_{tag}'])
    return uv_mat_to_np(u_mat, v_mat)


def read_convert_xy_mat(containing_dir):
    x_mat = np.array(sio.loadmat(f'{containing_dir}/x.mat')['x'])
    y_mat = np.array(sio.loadmat(f'{containing_dir}/y.mat')['y'])
    return np.dstack([x_mat[0][0], y_mat[0][0]]).astype(np.float64)


def read_and_export_uv_xy(containing_dir, tag):
    uv = read_convert_uv_mat(containing_dir, tag)
    np.save(f'{containing_dir}/vel_{tag}', uv)
    xy = read_convert_xy_mat(containing_dir)
    np.save(f'{containing_dir}/pos', xy)


read_and_export_uv_xy(sys.argv[1], sys.argv[2])
