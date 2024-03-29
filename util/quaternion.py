import numpy as np
from numpy import linalg as LA


def get_quat3(q):
    qx = q[..., 0]
    qy = q[..., 1]
    qz = q[..., 2]
    qw = q[..., 3]
    q3w = 1 - qx * qx - qz * qz
    q3x = qw * qx + qy * qz
    q3z = -qx * qy + qz * qw
    q3 = np.stack((q3x, q3z, q3w), axis=-1)
    return q3 / np.repeat(LA.norm(q3, axis=-1), 3).reshape(-1, 3)


def get_quat4(q3):
    q3x = q3[..., 0]
    q3z = q3[..., 1]
    q3w = q3[..., 2]

    qy = np.zeros_like(q3x)
    q = np.stack((q3x, qy, q3z, q3w), axis=-1)
    return q / np.repeat(LA.norm(q, axis=-1), 4).reshape(-1, 4)


def rotate_using_quaternion(v, q):
    qx = q[..., 0]
    qy = q[..., 1]
    qz = q[..., 2]
    qw = q[..., 3]
    rotated_x = (1 - 2 * (qy * qy + qz * qz)) * v[0] + 2 * (
        qx * qy - qz * qw) * v[1] + 2 * (qx * qz + qy * qw) * v[2]
    rotated_y = 2 * (qx * qy + qz * qw) * v[0] + (
        1 - 2 * (qx * qx + qz * qz)) * v[1] + 2 * (qy * qz - qx * qw) * v[2]
    rotated_z = 2 * (qx * qz - qy * qw) * v[0] + 2 * (
        qy * qz + qx * qw) * v[1] + (1 - 2 * (qx * qx + qy * qy)) * v[2]
    return np.stack((rotated_x, rotated_y, rotated_z), axis=-1)
