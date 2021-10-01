import numpy as np
import scipy.io as sio
import sys


def uv_mat_to_np(u_mat, v_mat, row_range):
    uv_np = np.zeros(
        [len(u_mat), (row_range[1] - row_range[0]),
         len(u_mat[0][0][0]), 2], np.float64)
    for time_id in range(len(u_mat)):
        uv_np[time_id] = uv = np.dstack([
            u_mat[time_id][0][row_range[0]:row_range[1]].astype(np.float64),
            v_mat[time_id][0][row_range[0]:row_range[1]].astype(np.float64)
        ])
    return uv_np


def read_convert_uv_mat(containing_dir, tag, row_range):
    u_mat = np.array(sio.loadmat(f'{containing_dir}/u_{tag}.mat')[f'u_{tag}'])
    v_mat = np.array(sio.loadmat(f'{containing_dir}/v_{tag}.mat')[f'v_{tag}'])
    return uv_mat_to_np(u_mat, v_mat, row_range)


def read_convert_xy_mat(containing_dir, row_range):
    x_mat = np.array(sio.loadmat(f'{containing_dir}/x.mat')['x'])
    y_mat = np.array(sio.loadmat(f'{containing_dir}/y.mat')['y'])
    return np.dstack([
        x_mat[0][0][row_range[0]:row_range[1]],
        y_mat[0][0][row_range[0]:row_range[1]]
    ]).astype(np.float64)


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


def read_and_export_uv_xy(containing_dir, tag):
    row_range = read_row_range(containing_dir)
    uv = read_convert_uv_mat(containing_dir, tag, row_range)
    np.save(f'{containing_dir}/vel_{tag}', uv)
    xy = read_convert_xy_mat(containing_dir, row_range)
    np.save(f'{containing_dir}/pos', xy)


read_and_export_uv_xy(sys.argv[1], sys.argv[2])
