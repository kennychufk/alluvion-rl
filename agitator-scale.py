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


agitator_options = [
    'bunny/bunny', '03797390/ec846432f3ebedf0a6f32a8797e3b9e9',
    '03046257/757fd88d3ddca2403406473757712946',
    '02942699/9db4b2c19c858a36eb34db531a289b8e',
    '03261776/1d4f9c324d6388a9b904f4192b538029',
    '03325088/daa26435e1603392b7a867e9b35a1295',
    '03759954/35f36e337df50fcb92b3c4741299c1af',
    '04401088/e5bb37415abcf3b7e1c1039f91f43fda',
    '04530566/66a90b7b92ff2549f2635cfccf45023',
    '03513137/91c0193d38f0c5338c9affdacaf55648',
    '03636649/83353863ea1349682ebeb1e6a8111f53'
]

for i, agitator_option in enumerate(agitator_options):
    agitator_mesh = al.Mesh()

    agitator_model_dir = f'{args.shape_dir}/{agitator_option}/models'
    agitator_mesh_filename = f'{agitator_model_dir}/manifold2-decimate-pa.obj'
    print(agitator_mesh_filename)
    print(Path(agitator_mesh_filename).is_file())
    agitator_mesh.set_obj(agitator_mesh_filename)
    agitator_original_vol, _, _, _ = agitator_mesh.calculate_mass_properties(1)
    print('agitator_original_vol', agitator_original_vol)
    # agitator_scale = np.cbrt(
    #     np.random.uniform(low=193, high=240) / agitator_original_vol)
    agitator_scale = np.cbrt(
        unit.from_real_volume(2.2e-5 +0.05e-5 * i) / agitator_original_vol)
    agitator_mesh.scale(agitator_scale)
    agitator_mesh.export_obj(f'{agitator_model_dir}/manifold2-decimate-2to-8.obj')
    new_vol, _, _, _ = agitator_mesh.calculate_mass_properties(1)
    print('new_vol real', unit.to_real_volume(new_vol))
    with open(f'{agitator_model_dir}/manifold2-decimate-2to-8.txt', 'w') as f:
        f.write(f'real volume = {unit.to_real_volume(new_vol)}')
