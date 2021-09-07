import glob
import argparse
import subprocess
import os
import numpy as np

for rmodel in np.arange(7.692, 8.2, 0.001):
    np.save('pipe_model_radius.npy', np.array([rmodel]))
    subprocess.Popen(
        ["python", "prefilled-hagen-poiseuille.py", ".alcache/9900.alu"],
        env=os.environ.copy()).wait()
