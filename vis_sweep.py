import glob
import argparse
import subprocess
import os
import numpy as np

# for kinematic_viscosity_real in np.arange(0.8e-6, 2e-5, 1e-7):
# for kinematic_viscosity_real in np.arange(0.8e-6, 2e-5, 1e-6):
kinematic_viscosity_real = 2.2e-5
np.save('kinematic_viscosity_real.npy', kinematic_viscosity_real)
subprocess.Popen(
    ["python", "prefilled-hagen-poiseuille.py", ".alcache/9900.alu"],
    env=os.environ.copy()).wait()
