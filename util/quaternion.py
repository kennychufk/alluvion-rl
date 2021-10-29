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
