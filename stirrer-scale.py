from pathlib import Path
import argparse
import math
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

import alluvion as al

from util import Unit, FluidSample, get_timestamp_and_hash, BuoySpec, parameterize_kinematic_viscosity

parser = argparse.ArgumentParser(description='agitator-scaling')
parser.add_argument('--shape-dir', type=str, required=True)
args = parser.parse_args()
dp = al.Depot(np.float32)

particle_radius = 0.25
kernel_radius = 1.0
density0 = 1.0
cubical_particle_volume = 8 * particle_radius * particle_radius * particle_radius
volume_relative_to_cube = 0.8
particle_mass = cubical_particle_volume * volume_relative_to_cube * density0

density0_real = np.random.uniform(low=990, high=1200)
unit = Unit(real_kernel_radius=2**-8,
            real_density0=density0_real,
            real_gravity=-9.80665)

agitator_options = ['stirrer/stirrer']

for i, agitator_option in enumerate(agitator_options):
    agitator_mesh = al.Mesh()

    agitator_model_dir = f'{args.shape_dir}/{agitator_option}/models'
    agitator_mesh_filename = f'{agitator_model_dir}/manifold2-decimate-pa.obj'
    print(agitator_mesh_filename)
    print(Path(agitator_mesh_filename).is_file())
    agitator_mesh.set_obj(agitator_mesh_filename)
    agitator_original_vol, _, _, _ = agitator_mesh.calculate_mass_properties(1)
    agitator_mesh.scale(256)
    agitator_mesh.export_obj(
        f'{agitator_model_dir}/manifold2-decimate-2to-8.obj')
    new_vol, _, _, _ = agitator_mesh.calculate_mass_properties(1)
    print('new_vol real', unit.to_real_volume(new_vol))
    with open(f'{agitator_model_dir}/manifold2-decimate-2to-8.txt', 'w') as f:
        f.write(f'real volume = {unit.to_real_volume(new_vol)}')
