import glob
import argparse
import subprocess
import os

import alluvion as al
import numpy as np
from numpy import linalg as LA
from pathlib import Path
import scipy.special as sc
from optim import AdamOptim

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import patheffects as path_effects


class OptimSampling:
    def __init__(self, dp, cn, pipe_length, pipe_radius, num_particles):
        self.num_sections = 14
        self.num_rotations = 16
        self.num_rs = 2048  # should be even number
        self.num_samples = self.num_rs * self.num_sections * self.num_rotations
        self.num_particles = num_particles

        # r contains 0 but not pipe_radius
        self.rs = pipe_radius / self.num_rs * np.arange(self.num_rs)

        self.sample_x = dp.create_coated((self.num_samples), 3)
        self.sample_density = dp.create_coated((self.num_samples), 1)
        self.sample_boundary = dp.create_coated(
            (dp.cni.num_boundaries, self.num_samples), 4)
        self.sample_boundary_kernel = dp.create_coated(
            (dp.cni.num_boundaries, self.num_samples), 4)
        self.sample_neighbors = dp.create_coated(
            (self.num_samples, dp.cni.max_num_neighbors_per_particle), 4)
        self.sample_num_neighbors = dp.create_coated((self.num_samples), 1,
                                                     np.uint32)
        sample_x_host = np.zeros((self.num_samples, 3), dp.default_dtype)
        section_length = pipe_length / self.num_sections
        offset_y_per_rotation = section_length / self.num_rotations
        theta_per_rotation = np.pi * 2 / self.num_rotations
        for i in range(self.num_samples):
            section_id = i // (self.num_rs * self.num_rotations)
            rotation_id = (i // self.num_rs) % (self.num_rotations)
            r_id = i % self.num_rs
            theta = theta_per_rotation * rotation_id
            sample_x_host[i] = np.array([
                self.rs[r_id] * np.cos(theta),
                pipe_length * -0.5 + section_length * section_id +
                offset_y_per_rotation * rotation_id,
                self.rs[r_id] * np.sin(theta)
            ], dp.default_dtype)
        self.sample_x.set(sample_x_host)
        self.dp = dp
        self.reset()

    def reset(self):
        self.density = np.zeros(self.num_rs, self.dp.default_dtype)
        self.bvol = np.zeros(self.num_rs, self.dp.default_dtype)
        self.bvolW = np.zeros(self.num_rs, self.dp.default_dtype)

    def aggregate(self):
        density_ungrouped = self.sample_density.get().reshape(-1, self.num_rs)
        self.density = np.mean(density_ungrouped, axis=0)
        bvol_ungrouped = self.sample_boundary.get().reshape(
            -1, self.num_rs, 4)[..., 3]
        self.bvol = np.mean(bvol_ungrouped, axis=0)
        bvolW_ungrouped = self.sample_boundary_kernel.get().reshape(
            -1, self.num_rs, 4)[..., 3]
        self.bvolW = np.mean(bvolW_ungrouped, axis=0)


def probe(dp, solver, osampling):
    solver.update_particle_neighbors_wrap1()
    runner.launch_make_neighbor_list_wrap1(osampling.sample_x, solver.pid,
                                           solver.pid_length,
                                           osampling.sample_neighbors,
                                           osampling.sample_num_neighbors,
                                           osampling.num_samples)
    solver.sample_all_boundaries(osampling.sample_x, osampling.sample_boundary,
                                 osampling.sample_boundary_kernel,
                                 osampling.num_samples)
    runner.launch_sample_density(osampling.sample_x,
                                 osampling.sample_neighbors,
                                 osampling.sample_num_neighbors,
                                 osampling.sample_density,
                                 osampling.sample_boundary_kernel,
                                 osampling.num_samples)
    osampling.aggregate()


dp = al.Depot(np.float64)
cn = dp.cn
cni = dp.cni
runner = dp.Runner()

# Physical constants
density0 = 1
dynamic_viscosity = 0.01

# Model parameters
scale_factor = 1
particle_radius = 0.25 * scale_factor
kernel_radius = particle_radius * 4
cubical_particle_volume = 8 * particle_radius * particle_radius * particle_radius
volume_relative_to_cube = 0.8
particle_mass = cubical_particle_volume * volume_relative_to_cube * density0

cn.set_cubic_discretization_constants()
cn.set_kernel_radius(kernel_radius)
cn.set_particle_attr(particle_radius, particle_mass, density0)

# Pipe dimension
pipe_radius_grid_span = 10
initial_radius = kernel_radius * pipe_radius_grid_span
pipe_model_radius = 7.772 * scale_factor  # 7.69324 (more stable at Re=1) for 9900 particles. Larger than the experimental value (7.69211562500019)
pipe_length_grid_half_span = 3
pipe_length_half = pipe_length_grid_half_span * kernel_radius
pipe_length = 2 * pipe_length_grid_half_span * kernel_radius
num_particles = dp.Runner.get_fluid_cylinder_num_particles(
    initial_radius, -pipe_length_half, pipe_length_half, particle_radius)
pipe_volume = num_particles * cn.particle_vol
pipe_radius = np.sqrt(pipe_volume / pipe_length / np.pi)
print('num_particles', num_particles)
print('pipe_radius', pipe_radius)

# rigids
max_num_contacts = 512
pile = dp.Pile(dp, runner, max_num_contacts)
pile.add(dp.InfiniteCylinderDistance.create(pipe_model_radius),
         al.uint3(512, 1, 512),
         sign=-1)

pile.build_grids(2 * kernel_radius)
pile.reallocate_kinematics_on_device()
cn.contact_tolerance = particle_radius

# grid
grid_res = al.uint3(int(pipe_radius_grid_span * 2),
                    int(pipe_length_grid_half_span * 2),
                    int(pipe_radius_grid_span * 2))
grid_offset = al.int3(-pipe_radius_grid_span, -pipe_length_grid_half_span,
                      -pipe_radius_grid_span)
cni.grid_res = grid_res
cni.grid_offset = grid_offset
cni.max_num_particles_per_cell = 64
cni.max_num_neighbors_per_particle = 64
cn.set_wrap_length(grid_res.y * kernel_radius)

solver = dp.SolverDf(runner,
                     pile,
                     dp,
                     num_particles,
                     grid_res,
                     enable_surface_tension=False,
                     enable_vorticity=False,
                     graphical=False)
solver.particle_radius = particle_radius
solver.num_particles = num_particles
solver.initial_dt = 1e-2
solver.max_dt = 1.0
solver.min_dt = 0
solver.cfl = 0.2

dp.copy_cn()

initial_particle_x = dp.create((num_particles), 3)
initial_particle_x.read_file('.alcache/9900.alu')
initial_particle_x.scale(dp.f3(scale_factor, scale_factor, scale_factor))
solver.particle_x.set_from(initial_particle_x)
solver.num_particles = initial_particle_x.get_shape()[0]

osampling = OptimSampling(dp, cn, pipe_length, pipe_radius,
                          solver.num_particles)
probe(dp, solver, osampling)
print(pipe_model_radius, osampling.bvol[-10:])

model_radius_list = np.arange(7.7, 7.8, 0.02)
np.save('rs.npy', osampling.rs)
np.save('model_radius_list.npy', model_radius_list)
bvol_array = np.zeros((len(model_radius_list), len(osampling.rs)),
                      dp.default_dtype)
for i, new_model_radius in enumerate(model_radius_list):
    pile.replace(0,
                 dp.InfiniteCylinderDistance.create(new_model_radius),
                 al.uint3(512, 1, 512),
                 sign=-1)
    pile.build_grids(2 * kernel_radius)
    pile.reallocate_kinematics_on_device()
    probe(dp, solver, osampling)
    print(new_model_radius, osampling.bvol[-10:])
    bvol_array[i] = osampling.bvol
np.save('bvol_array', bvol_array)
