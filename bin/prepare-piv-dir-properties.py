import numpy as np
from util import read_file_float

piv_truth_dirs = [
    '/media/kennychufk/vol1bk0/20210415_162749-laser-too-high/',
    '/media/kennychufk/vol1bk0/20210415_164304/',
    '/media/kennychufk/vol1bk0/20210416_101435/',
    '/media/kennychufk/vol1bk0/20210416_102548/',
    '/media/kennychufk/vol1bk0/20210416_103739/',
    '/media/kennychufk/vol1bk0/20210416_104936/',
    '/media/kennychufk/vol1bk0/20210416_120534/',
    '/media/kennychufk/vol1bk0/20210416_114327',  # 8
    '/media/kennychufk/vol1bk0/20210416_115523',  # 9
    # freehand
    '/media/kennychufk/vol1bk0/20210416_154851',  # water
    '/media/kennychufk/vol1bk0/20210416_155948',  # water
    '/media/kennychufk/vol1bk0/20210416_161703',  # water
    '/media/kennychufk/vol1bk0/20210416_162740',  # water
    '/media/kennychufk/vol1bk0/20210416_164544-slightly-wrong-focus',  # gwm
    '/media/kennychufk/vol1bk0/20210416_165720',  # gwm
]

for truth_dir in piv_truth_dirs:
    density0_real = read_file_float(f'{truth_dir}/density.txt')
    print('density0_real', density0_real)
    np.save(f'{truth_dir}/density0_real.npy', density0_real)

    kinematic_viscosity_real = read_file_float(
        f'{truth_dir}/dynamic_viscosity.txt') / density0_real
    print('kinematic_viscosity_real', kinematic_viscosity_real)
    np.save(f'{truth_dir}/kinematic_viscosity_real.npy',
            kinematic_viscosity_real)

    fluid_mass = read_file_float(f'{truth_dir}/mass.txt')
    print('fluid_mass', fluid_mass)
    np.save(f'{truth_dir}/fluid_mass.npy', fluid_mass)
