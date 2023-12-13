from pathlib import Path
import argparse
import math
import time
import numpy as np
import subprocess
import os
from scipy.spatial.transform import Rotation as R
from PIL import Image

import alluvion as al

from util import Unit, FluidSamplePellets, get_timestamp_and_hash, BuoySpec, parameterize_kinematic_viscosity

parser = argparse.ArgumentParser(description='Initial bead generator')
parser.add_argument('--existing', type=str, default='')
parser.add_argument('--kernel-radius', type=float, default=0.015625)
parser.add_argument('--density0', type=float, default=1000)
parser.add_argument('--fluid-mass', type=float, required=True)
parser.add_argument('--cache-dir', type=str, default='cache')
args = parser.parse_args()
dp = al.Depot(np.float32)
cn = dp.cn
cni = dp.cni
dp.create_display(800, 600, "alluvion-fixed", False)
display_proxy = dp.get_display_proxy()
framebuffer = display_proxy.create_framebuffer()
runner = dp.Runner()

particle_radius = 0.25
kernel_radius = 1.0
density0 = 1.0
cubical_particle_volume = 8 * particle_radius * particle_radius * particle_radius
volume_relative_to_cube = 0.8
particle_mass = cubical_particle_volume * volume_relative_to_cube * density0

gravity = dp.f3(0, -1, 0)

density0_real = args.density0
real_kernel_radius = args.kernel_radius
unit = Unit(real_kernel_radius=real_kernel_radius,
            real_density0=density0_real,
            real_gravity=-9.80665)

cn.set_kernel_radius(kernel_radius)
cn.set_particle_attr(particle_radius, particle_mass, density0)
cn.boundary_epsilon = 1e-9
cn.gravity = gravity
kinematic_viscosity_real = 1e-6  # TODO: discrete, either water or GWM
cn.viscosity, cn.boundary_viscosity = unit.from_real_kinematic_viscosity(
    parameterize_kinematic_viscosity(kinematic_viscosity_real))
print('parameterized nu',
      parameterize_kinematic_viscosity(kinematic_viscosity_real)[0])
cni.max_num_particles_per_cell = 64
cni.max_num_neighbors_per_particle = 64

container_pellet_filename = f'geometry-data/cube24-{real_kernel_radius}.alu'  # TODO: use shape_dir

container_num_pellets = dp.get_alu_info(container_pellet_filename)[0][0]
num_pellets = container_num_pellets

# rigids
max_num_contacts = 512
pile = dp.Pile(dp, runner, max_num_contacts, al.VolumeMethod.pellets,
               num_pellets)

target_container_volume = unit.from_real_volume(0.008)
container_mesh = al.Mesh()

## ================== using cube
container_width = unit.from_real_length(0.24)
container_dim = dp.f3(container_width, container_width, container_width)
container_mesh = al.Mesh()
container_mesh.set_box(container_dim, 8)
container_distance = dp.BoxDistance.create(container_dim, outset=0)
container_extent = container_distance.aabb_max - container_distance.aabb_min
container_res_float = container_extent / particle_radius
container_res = al.uint3(int(container_res_float.x),
                         int(container_res_float.y),
                         int(container_res_float.z))
print('container_res', container_res)
container_pellet_x = dp.create((container_num_pellets), 3)
container_pellet_x.read_file(container_pellet_filename)
pile.add_pellets(container_distance,
                 container_res,
                 pellets=container_pellet_x,
                 sign=-1,
                 mass=0,
                 restitution=0.8,
                 friction=0.3)
dp.remove(container_pellet_x)
## ================== using cube

pile.reallocate_kinematics_on_device()
pile.set_gravity(gravity)
cn.contact_tolerance = particle_radius * 2

fluid_block_mode = 0
use_existing = len(args.existing) > 0
existing_filename = args.existing
if not use_existing:
    num_positions = dp.Runner.get_fluid_block_num_particles(
        mode=fluid_block_mode,
        box_min=container_distance.aabb_min,
        box_max=container_distance.aabb_max,
        particle_radius=particle_radius)
    internal_encoded = dp.create_coated((num_positions), 1, np.uint32)
    max_fill_num_particles = pile.compute_sort_fluid_block_internal_all(
        internal_encoded,
        box_min=container_distance.aabb_min,
        box_max=container_distance.aabb_max,
        particle_radius=particle_radius,
        mode=fluid_block_mode)
num_particles = int(args.fluid_mass / unit.to_real_mass(particle_mass))

print('num_particles', num_particles)

container_aabb_range_per_h = container_extent / kernel_radius
cni.grid_res = al.uint3(int(math.ceil(container_aabb_range_per_h.x)),
                        int(math.ceil(container_aabb_range_per_h.y)),
                        int(math.ceil(container_aabb_range_per_h.z))) + 4
cni.grid_offset = al.int3(
    int(container_distance.aabb_min.x) - 2,
    int(container_distance.aabb_min.y) - 2,
    int(container_distance.aabb_min.z) - 2)
print('grid_res', cni.grid_res)
print('grid_offset', cni.grid_offset)

solver = dp.SolverI(runner,
                    pile,
                    dp,
                    num_particles,
                    enable_surface_tension=False,
                    enable_vorticity=False,
                    graphical=True)
particle_normalized_attr = dp.create_graphical_like(solver.particle_density)

solver.num_particles = num_particles
solver.max_dt = unit.from_real_time(0.00025)
solver.initial_dt = solver.max_dt
solver.min_dt = 0
solver.cfl = 0.2
solver.min_density_solve = 5

dp.map_graphical_pointers()
if use_existing:
    solver.particle_x.read_file(existing_filename)
else:
    runner.launch_create_fluid_block_internal(
        solver.particle_x,
        internal_encoded,
        num_particles,
        offset=0,
        particle_radius=particle_radius,
        mode=fluid_block_mode,
        box_min=container_distance.aabb_min,
        box_max=container_distance.aabb_max)
    dp.remove(internal_encoded)
dp.unmap_graphical_pointers()

display_proxy.set_camera(unit.from_real_length(al.float3(0, 0.6, 0.6)),
                         unit.from_real_length(al.float3(0, -0.02, 0)))
display_proxy.set_clip_planes(unit.to_real_length(1),
                              container_distance.aabb_max.z * 20)
colormap_tex = display_proxy.create_colormap_viridis()

display_proxy.add_bind_framebuffer_step(framebuffer)
display_proxy.add_particle_shading_program(solver.particle_x,
                                           particle_normalized_attr,
                                           colormap_tex,
                                           solver.particle_radius, solver)
display_proxy.add_pile_shading_program(pile)
display_proxy.add_show_framebuffer_shader(framebuffer)
display_proxy.resize(800, 600)

with open('switch', 'w') as f:
    f.write('1')
while True and unit.to_real_time(solver.t) < 60:
    with open('switch') as f:
        switch_result = int(f.read())
        if switch_result == 0:
            break
    dp.map_graphical_pointers()
    for frame_interstep in range(200):
        solver.step()
    v_rms = np.sqrt(
        runner.sum(solver.particle_cfl_v2, solver.num_particles) /
        solver.num_particles)
    print(
        f"t = {unit.to_real_time(solver.t) } dt = {unit.to_real_time(solver.dt)} cfl = {solver.utilized_cfl} vrms={unit.to_real_velocity(v_rms)} max_v={unit.to_real_velocity(np.sqrt(solver.max_v2))} num solves = {solver.num_density_solve}"
    )
    solver.normalize(solver.particle_v, particle_normalized_attr, 0,
                     unit.from_real_velocity(0.01))
    dp.unmap_graphical_pointers()
    display_proxy.draw()

dp.map_graphical_pointers()
solver.particle_x.write_file(
    f"{args.cache_dir}/playground_bead_x-{real_kernel_radius}-{density0_real}-{args.fluid_mass}.alu",
    solver.num_particles)
solver.particle_v.write_file(
    f"{args.cache_dir}/playground_bead_v-{real_kernel_radius}-{density0_real}-{args.fluid_mass}.alu",
    solver.num_particles)
solver.particle_pressure.write_file(
    f"{args.cache_dir}/playground_bead_p-{real_kernel_radius}-{density0_real}-{args.fluid_mass}.alu",
    solver.num_particles)
dp.unmap_graphical_pointers()

dp.remove(particle_normalized_attr)
del solver
del pile
del runner
del dp
