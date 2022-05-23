import glob
import argparse
import subprocess
import os
import numpy as np

# for kinematic_viscosity_real in np.arange(0.5e-6, 2.1e-5, 0.5e-6):
for kinematic_viscosity_real in np.arange(7.5e-6, 2.1e-5, 0.5e-6):
    np.save('kinematic_viscosity_real.npy', kinematic_viscosity_real)
    subprocess.Popen([
        "python", "prefilled-hagen-poiseuille-pellets.py",
        ".alcache/shrink-bak/annealing-best-9900.alu",
        ".alcache/shrink-bak/annealing-best-9900-pellets.alu",
        ".alcache/shrink-bak/shrink-best-stat9900.npy"
    ],
                     env=os.environ.copy()).wait()
