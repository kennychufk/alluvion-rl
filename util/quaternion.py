import numpy as np


def get_quat3(q):
    qx = q[..., 0]
    qy = q[..., 1]
    qz = q[..., 2]
    qw = q[..., 3]
    q3w = np.sqrt(1 - qx * qx - qz * qz)
    q3x = (qw * qx + qy * qz) / q3w
    q3z = (-qx * qy + qz * qw) / q3w
    return np.stack((q3x, q3z, q3w), axis=-1)


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
