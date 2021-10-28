import numpy as np


def parameterize_kinematic_viscosity(nu):
    return np.array([2.0761830957036738, 6.544046942658794]) * nu + np.array(
        [1.0647339827434552e-09, -2.429370995158124e-08])
