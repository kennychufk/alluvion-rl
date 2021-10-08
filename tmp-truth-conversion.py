from pathlib import Path
import math
import time
import numpy as np
import xxhash
import glob
import sys

import alluvion as al

from util import Unit, FluidSample

dp = al.Depot(np.float32)
cn = dp.cn
cni = dp.cni
runner = dp.Runner()

particle_radius = 0.25
kernel_radius = 1.0
density0 = 1.0
cubical_particle_volume = 8 * particle_radius * particle_radius * particle_radius
volume_relative_to_cube = 0.8
particle_mass = cubical_particle_volume * volume_relative_to_cube * density0

gravity = dp.f3(0, -1, 0)

# real_kernel_radius = 0.0025
unit = Unit(real_kernel_radius=0.005,
            real_density0=1000,
            real_gravity=-9.80665)

containing_dir = sys.argv[1]

# sample_shape = dp.get_alu_info(f'{containing_dir}/v-0.alu')[0]
# particle_shape = dp.get_alu_info(f'{containing_dir}/visual-x-0.alu')[0]
# sample_data3 = dp.create_coated(sample_shape, 3)
# sample_data1 = dp.create_coated(sample_shape, 1)
# particle_x = dp.create_coated(particle_shape, 3)
# 
# to_real_length_scale = unit.to_real_length(1)
# to_real_length_scale3 = dp.f3(to_real_length_scale, to_real_length_scale,
#                               to_real_length_scale)
# # sample
# container_width = unit.from_real_length(0.24)
# num_samples_per_dim = 24
# num_samples = num_samples_per_dim * num_samples_per_dim * num_samples_per_dim
# sample_x_np = np.zeros((num_samples, 3), dp.default_dtype)
# samle_box_min = dp.f3(container_width * -0.5, 0, container_width * -0.5)
# samle_box_max = dp.f3(container_width * 0.5, container_width,
#                       container_width * 0.5)
# samle_box_size = samle_box_max - samle_box_min
# for i in range(num_samples):
#     z_id = i % num_samples_per_dim
#     y_id = i % (num_samples_per_dim *
#                 num_samples_per_dim) // num_samples_per_dim
#     x_id = i // (num_samples_per_dim * num_samples_per_dim)
#     sample_x_np[i] = np.array([
#         samle_box_min.x + samle_box_size.x / (num_samples_per_dim - 1) * x_id,
#         samle_box_min.y + samle_box_size.y / (num_samples_per_dim - 1) * y_id,
#         samle_box_min.z + samle_box_size.z / (num_samples_per_dim - 1) * z_id
#     ])
# sampling = FluidSample(dp, sample_x_np)
# sampling.sample_x.scale(to_real_length_scale3)
# sampling.sample_x.write_file(f'{containing_dir}/sample-x.alu', num_samples)
# # sample
# 
# to_real_velocity_scale = unit.to_real_velocity(1)
# to_real_velocity_scale3 = dp.f3(to_real_velocity_scale, to_real_velocity_scale,
#                                 to_real_velocity_scale)
# vfield_filepaths = glob.glob(f'{containing_dir}/v-*.alu')
# for vfield_filepath in vfield_filepaths:
#     sample_data3.read_file(vfield_filepath)
#     sample_data3.scale(to_real_velocity_scale3)
#     sample_data3.write_file(vfield_filepath, sample_shape[0])
# 
# to_real_density_scale = unit.to_real_density(1)
# densityfield_filepaths = glob.glob(f'{containing_dir}/density-*.alu')
# for densityfield_filepath in densityfield_filepaths:
#     sample_data1.read_file(densityfield_filepath)
#     sample_data1.scale(to_real_density_scale)
#     sample_data1.write_file(densityfield_filepath, sample_shape[0])
# 
# visualx_filepaths = glob.glob(f'{containing_dir}/visual-x-*.alu')
# for visualx_filepath in visualx_filepaths:
#     particle_x.read_file(visualx_filepath)
#     particle_x.scale(to_real_length_scale3)
#     particle_x.write_file(visualx_filepath, particle_shape[0])
# 
pile = dp.Pile(dp, runner, 1)
dummy_dim = dp.f3(1, 1, 1)
dummy_mesh = al.Mesh()
dummy_mesh.set_box(dummy_dim, 8)
for i in range(8 + 2):
    pile.add(dp.BoxDistance.create(dummy_dim),
             al.uint3(4, 4, 4),
             collision_mesh=dummy_mesh,
             mass=0,
             restitution=0.8,
             friction=0.2,
             display_mesh=al.Mesh())

# to_real_angular_velocity_scale = unit.to_real_angular_velocity(1)
pile_filepaths = glob.glob(f'{containing_dir}/*.pile')
for pile_filepath in pile_filepaths:
    if "visual" not in pile_filepath:
        pile.read_file(pile_filepath)
        pile.write_file(pile_filepath, unit.to_real_length(1), unit.to_real_velocity(1), unit.to_real_angular_velocity(1))
