import numpy as np


def parameterize_kinematic_viscosity(nu):
    return np.array([2.0761830957036738, 6.544046942658794]) * nu + np.array(
        [1.0647339827434552e-09, -2.429370995158124e-08])


def parameterize_kinematic_viscosity_with_pellets(nu):
    return np.array([2.075526047059374, 4.024908551888513]) * nu + np.array(
        [1.6769618666249118e-12, -4.9294025838194505e-11])
