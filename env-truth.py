from pathlib import Path
import argparse
import math
import time
import numpy as np

import alluvion as al

from util import Unit, FluidSample, get_timestamp_and_hash

parser = argparse.ArgumentParser(description='RL ground truth generator')
parser.add_argument('--initial', type=str, default='')
parser.add_argument('--output-dir', type=str, default='.')
args = parser.parse_args()
dp = al.Depot(np.float32)
cn = dp.cn
cni = dp.cni
dp.create_display(800, 600, "", False)
display_proxy = dp.get_display_proxy()
runner = dp.Runner()

particle_radius = 0.25
kernel_radius = 1.0
density0 = 1.0
cubical_particle_volume = 8 * particle_radius * particle_radius * particle_radius
volume_relative_to_cube = 0.8
particle_mass = cubical_particle_volume * volume_relative_to_cube * density0

gravity = dp.f3(0, -1, 0)

unit = Unit(real_kernel_radius=0.005,
            real_density0=1000,
            real_gravity=-9.80665)

cn.set_cubic_discretization_constants()
cn.set_kernel_radius(kernel_radius)
cn.set_particle_attr(particle_radius, particle_mass, density0)
cn.boundary_epsilon = 1e-9
cn.gravity = gravity
cn.viscosity, cn.boundary_viscosity = unit.from_real_kinematic_viscosity(
    np.array([2.049e-6, 6.532e-6]))

# rigids
max_num_contacts = 512
pile = dp.Pile(dp, runner, max_num_contacts)

container_width = unit.from_real_length(0.24)
container_dim = dp.f3(container_width, container_width, container_width)
container_mesh = al.Mesh()
container_mesh.set_box(container_dim, 8)
container_distance = dp.BoxDistance.create(container_dim, outset=0.46153312)
pile.add(container_distance,
         al.uint3(64, 64, 64),
         sign=-1,
         thickness=kernel_radius,
         collision_mesh=container_mesh,
         mass=0,
         restitution=0.8,
         friction=0.2,
         inertia_tensor=dp.f3(1, 1, 1),
         x=dp.f3(0, container_width / 2, 0),
         q=dp.f4(0, 0, 0, 1),
         display_mesh=al.Mesh())

current_inset = 0.403

cylinder_radius = unit.from_real_length(3.0088549658278843e-3)
cylinder_height = unit.from_real_length(38.5e-3)
cylinder_mass = unit.from_real_mass(1.06e-3)  # TODO: randomize
cylinder_comy = unit.from_real_length(-8.852102803738316e-3)
cylinder_volume = cylinder_radius * cylinder_radius * np.pi * cylinder_height
cylinder_neutral_buoyant_force = -cylinder_volume * density0 * gravity.y
cylinder_mesh = al.Mesh()
cylinder_mesh.set_cylinder(cylinder_radius, cylinder_height, 24, 24)
cylinder_mesh.translate(dp.f3(0, -cylinder_comy, 0))
cylinder_inertia = unit.from_real_moment_of_inertia(
    dp.f3(7.911343969145678e-8, 2.944622178863632e-8, 7.911343969145678e-8))
cylinder_map_radial_size = 24
cylinder_map_dim = al.uint3(
    cylinder_map_radial_size,
    int(cylinder_map_radial_size * cylinder_height / cylinder_radius / 2),
    cylinder_map_radial_size)

num_buoys = 8
for i in range(num_buoys):
    pile.add(dp.CylinderDistance.create(cylinder_radius - current_inset,
                                        cylinder_height - current_inset * 2,
                                        cylinder_comy),
             cylinder_map_dim,
             sign=1,
             collision_mesh=cylinder_mesh,
             mass=cylinder_mass,
             restitution=0.3,
             friction=0.4,
             inertia_tensor=cylinder_inertia,
             x=dp.f3(
                 np.random.uniform(-container_width * 0.45,
                                   container_width * 0.45),
                 container_width * np.random.uniform(0.55, 0.75),
                 np.random.uniform(-container_width * 0.45,
                                   container_width * 0.45)),
             q=dp.f4(0, 0, 0, 1),
             display_mesh=cylinder_mesh)

generating_initial = (len(args.initial) == 0)

if not generating_initial:
    bunny_mesh = al.Mesh()
    bunny_filename = '3dmodels/bunny-pa.obj'
    bunny_mesh.set_obj(bunny_filename)
    bunny_max_radius = unit.from_real_length(0.04)
    bunny_mesh.scale(bunny_max_radius)
    bunny_density = unit.from_real_density(800)
    bunny_mass, bunny_com, bunny_inertia, bunny_inertia_off_diag = bunny_mesh.calculate_mass_properties(
        bunny_density)
    bunny_triangle_mesh = dp.TriangleMesh()
    bunny_mesh.copy_to(bunny_triangle_mesh)
    bunny_distance = dp.MeshDistance.create(bunny_triangle_mesh)
    bunny_xz_half_range = container_width * 0.5 - bunny_max_radius - kernel_radius
    bunny_id = pile.add(bunny_distance,
                        al.uint3(40, 40, 40),
                        sign=1,
                        collision_mesh=bunny_mesh,
                        mass=bunny_mass,
                        restitution=0.8,
                        friction=0.3,
                        inertia_tensor=bunny_inertia,
                        x=dp.f3(
                            np.random.uniform(low=-bunny_xz_half_range,
                                              high=bunny_xz_half_range),
                            container_width * 0.29,
                            np.random.uniform(low=-bunny_xz_half_range,
                                              high=bunny_xz_half_range)),
                        q=dp.f4(0, 0, 0, 1),
                        display_mesh=bunny_mesh)

pile.build_grids(2 * kernel_radius)
pile.reallocate_kinematics_on_device()
pile.set_gravity(gravity)
cn.contact_tolerance = particle_radius

block_mode = 0
edge_factor = 0.49
box_min = dp.f3((container_width - 2 * kernel_radius) * -edge_factor,
                kernel_radius,
                (container_width - kernel_radius * 2) * -edge_factor)
box_max = dp.f3((container_width - 2 * kernel_radius) * edge_factor,
                container_width * 0.32,
                (container_width - kernel_radius * 2) * edge_factor)
num_particles = dp.Runner.get_fluid_block_num_particles(
    mode=block_mode,
    box_min=box_min,
    box_max=box_max,
    particle_radius=particle_radius)
print('num_particles', num_particles)
container_aabb_range = container_distance.aabb_max - container_distance.aabb_min
container_aabb_range_per_h = container_aabb_range / kernel_radius
cni.grid_res = al.uint3(int(math.ceil(container_aabb_range_per_h.x)),
                        int(math.ceil(container_aabb_range_per_h.y)),
                        int(math.ceil(container_aabb_range_per_h.z))) + 4
cni.grid_offset = al.int3(
    -(int(cni.grid_res.x) // 2) - 2,
    -int(math.ceil(container_distance.outset / kernel_radius)) - 1,
    -(int(cni.grid_res.z) // 2) - 2)
cni.max_num_particles_per_cell = 64
cni.max_num_neighbors_per_particle = 64

solver = dp.SolverI(runner,
                    pile,
                    dp,
                    num_particles,
                    cni.grid_res,
                    enable_surface_tension=False,
                    enable_vorticity=False,
                    graphical=True)
particle_normalized_attr = dp.create_graphical_like(solver.particle_density)

solver.num_particles = num_particles
solver.max_dt = unit.from_real_time(0.1 * unit.rl)
solver.initial_dt = solver.max_dt
solver.min_dt = 0
solver.cfl = 0.4

dp.copy_cn()

dp.map_graphical_pointers()

if generating_initial:
    runner.launch_create_fluid_block(solver.particle_x,
                                     solver.num_particles,
                                     offset=0,
                                     mode=block_mode,
                                     box_min=box_min,
                                     box_max=box_max)
else:
    solver.particle_x.read_file(f'{args.initial}/x.alu')
    solver.particle_v.read_file(f'{args.initial}/v.alu')
    solver.particle_pressure.read_file(f'{args.initial}/pressure.alu')
    pile.read_file(f'{args.initial}/container_buoys.pile', num_buoys + 1)
dp.unmap_graphical_pointers()
display_proxy.set_camera(unit.from_real_length(al.float3(0, 0.06, 0.4)),
                         unit.from_real_length(al.float3(0, 0.06, 0)))
display_proxy.set_clip_planes(unit.to_real_length(1), box_max.z * 20)
colormap_tex = display_proxy.create_colormap_viridis()

display_proxy.add_particle_shading_program(solver.particle_x,
                                           particle_normalized_attr,
                                           colormap_tex,
                                           solver.particle_radius, solver)
display_proxy.add_pile_shading_program(pile)

next_force_time = 0.0
remaining_force_time = 0.0

timestamp_str, timestamp_hash = get_timestamp_and_hash()
if generating_initial:
    initial_directory = f'{args.output_dir}/rlinit-{timestamp_hash}-{timestamp_str}'
    Path(initial_directory).mkdir(parents=True, exist_ok=True)
else:
    frame_directory = f'{args.output_dir}/rltruth-{timestamp_hash}-{timestamp_str}'
    Path(frame_directory).mkdir(parents=True, exist_ok=True)
    with open(f'{frame_directory}/init.txt', 'w') as f:
        f.write(args.initial)

# Sampling
num_samples_per_dim = 24
num_samples = num_samples_per_dim * num_samples_per_dim * num_samples_per_dim
sample_x_np = np.zeros((num_samples, 3), dp.default_dtype)

samle_box_min = dp.f3(container_width * -0.5, 0, container_width * -0.5)
samle_box_max = dp.f3(container_width * 0.5, container_width,
                      container_width * 0.5)
samle_box_size = samle_box_max - samle_box_min
for i in range(num_samples):
    z_id = i % num_samples_per_dim
    y_id = i % (num_samples_per_dim *
                num_samples_per_dim) // num_samples_per_dim
    x_id = i // (num_samples_per_dim * num_samples_per_dim)
    sample_x_np[i] = np.array([
        samle_box_min.x + samle_box_size.x / (num_samples_per_dim - 1) * x_id,
        samle_box_min.y + samle_box_size.y / (num_samples_per_dim - 1) * y_id,
        samle_box_min.z + samle_box_size.z / (num_samples_per_dim - 1) * z_id
    ])
sampling = FluidSample(dp, sample_x_np)
if not generating_initial:
    real_sample_x = dp.create_coated_like(sampling.sample_x)
    real_sample_x.set_from(sampling.sample_x)
    real_sample_x.scale(unit.to_real_length(1))
    real_sample_x.write_file(f'{frame_directory}/sample-x.alu',
                             sampling.num_samples)
    dp.remove(real_sample_x)

truth_real_freq = 100.0
truth_real_interval = 1.0 / truth_real_freq
next_truth_frame_id = 0

visual_real_freq = 30.0
visual_real_interval = 1.0 / visual_real_freq
next_visual_frame_id = 0
visual_x_scaled = dp.create_coated_like(solver.particle_x)

target_t = unit.from_real_time(10.0)

with open('switch', 'w') as f:
    f.write('1')
while generating_initial or solver.t < target_t:
    dp.map_graphical_pointers()
    if generating_initial:
        with open('switch', 'r') as f:
            if f.read(1) == '0':
                solver.particle_x.write_file(f'{initial_directory}/x.alu',
                                             solver.num_particles)
                solver.particle_v.write_file(f'{initial_directory}/v.alu',
                                             solver.num_particles)
                solver.particle_pressure.write_file(
                    f'{initial_directory}/pressure.alu', solver.num_particles)
                pile.write_file(f'{initial_directory}/container_buoys.pile')
                break
    for frame_interstep in range(10):
        if not generating_initial:
            if solver.t >= unit.from_real_time(
                    next_truth_frame_id * truth_real_interval):
                sampling.prepare_neighbor_and_boundary(runner, solver)
                sample_v = sampling.sample_velocity(runner, solver)
                sample_v.scale(unit.to_real_velocity(1))
                sample_v.write_file(
                    f'{frame_directory}/v-{next_truth_frame_id}.alu',
                    sampling.num_samples)
                sample_density = sampling.sample_density(runner)
                sample_density.scale(unit.to_real_density(1))
                sample_density.write_file(
                    f'{frame_directory}/density-{next_truth_frame_id}.alu',
                    sampling.num_samples)
                pile.write_file(
                    f'{frame_directory}/{next_truth_frame_id}.pile',
                    unit.to_real_length(1), unit.to_real_velocity(1),
                    unit.to_real_angular_velocity(1))
                next_truth_frame_id += 1
            if solver.t >= unit.from_real_time(
                    next_visual_frame_id * visual_real_interval):
                visual_x_scaled.set_from(solver.particle_x)
                visual_x_scaled.scale(unit.to_real_length(1))
                visual_x_scaled.write_file(
                    f'{frame_directory}/visual-x-{next_visual_frame_id}.alu',
                    solver.num_particles)
                pile.write_file(
                    f'{frame_directory}/visual-{next_visual_frame_id}.pile',
                    unit.to_real_length(1), unit.to_real_velocity(1),
                    unit.to_real_angular_velocity(1))
                next_visual_frame_id += 1
            ###### move object #########
            bunny_v_al = pile.v[bunny_id]
            bunny_omega_al = pile.omega[bunny_id]
            bunny_v = np.array([bunny_v_al.x, bunny_v_al.y, bunny_v_al.z],
                               dp.default_dtype)
            bunny_omega = np.array(
                [bunny_omega_al.x, bunny_omega_al.y, bunny_omega_al.z],
                dp.default_dtype)

            if remaining_force_time <= 0 and next_force_time < 0:
                next_force_time = solver.t + unit.from_real_time(
                    np.random.rand(1) * 0.12)
                print('next_force_time', next_force_time)
            if next_force_time <= solver.t and remaining_force_time <= 0:
                next_force_time = -1
                remaining_force_time = unit.from_real_time(
                    np.random.rand(1) * 0.8)
                bunny_x = pile.x[bunny_id]
                bunny_angular_acc = unit.from_real_angular_acceleration(
                    (np.random.rand(3) - 0.5) * 20.1 * np.pi)
                bunny_a = unit.from_real_acceleration(
                    np.random.uniform(low=1.0, high=5.0, size=3))
                bunny_a[0] *= -(np.sign(bunny_x.x))
                bunny_a[1] = unit.from_real_acceleration(
                    (np.random.rand(1) - 0.7) * 2.0)
                bunny_a[2] *= -(np.sign(bunny_x.z))
                print('start force', bunny_a, bunny_angular_acc)

            if remaining_force_time > 0:
                bunny_v += bunny_a * solver.dt
                bunny_omega += bunny_angular_acc * solver.dt

                pile.v[bunny_id] = dp.f3(bunny_v[0], bunny_v[1], bunny_v[2])
                pile.omega[bunny_id] = dp.f3(bunny_omega[0], bunny_omega[1],
                                             bunny_omega[2])
                remaining_force_time -= solver.dt
            ###### move object #########
        solver.step()
    print(
        f"t = {unit.to_real_time(solver.t) } dt = {unit.to_real_time(solver.dt)} cfl = {solver.utilized_cfl} num solves = {solver.num_density_solve}"
    )

    solver.normalize(solver.particle_v, particle_normalized_attr, 0,
                     unit.from_real_velocity(0.02))
    dp.unmap_graphical_pointers()
    display_proxy.draw()
