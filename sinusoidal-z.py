import sys
import argparse
from pathlib import Path

import numpy as np
import alluvion as al

from util import Unit, FluidSample, parameterize_kinematic_viscosity, BuoySpec, RigidInterpolator
from util import get_timestamp_and_hash
from util import read_file_int, read_file_float

parser = argparse.ArgumentParser(description='Compare with PIV truth')
parser.add_argument('--truth-dir', type=str, required=True)
parser.add_argument('--display', metavar='d', type=bool, default=False)
args = parser.parse_args()

dp = al.Depot(np.float32)
cn = dp.cn
cni = dp.cni

if args.display:
    dp.create_display(800, 600, "", False)

display_proxy = dp.get_display_proxy() if args.display else None
runner = dp.Runner()

particle_radius = 0.25
spacing = 0.259967967091019859280982201077
kernel_radius = 1.0
density0 = 1.0
cubical_particle_volume = 8 * particle_radius * particle_radius * particle_radius
volume_relative_to_cube = 0.8
particle_mass = cubical_particle_volume * volume_relative_to_cube * density0

gravity = dp.f3(0, -1, 0)

# real_kernel_radius = 0.005
real_kernel_radius = 0.010
unit = Unit(real_kernel_radius=real_kernel_radius,
            real_density0=read_file_float(f'{args.truth_dir}/density.txt'),
            real_gravity=-9.80665)

cn.set_kernel_radius(kernel_radius)
cn.set_particle_attr(particle_radius, particle_mass, density0)
cn.boundary_epsilon = 1e-9
cn.gravity = gravity
kinematic_viscosity_real = read_file_float(
    f'{args.truth_dir}/dynamic_viscosity.txt') / unit.rdensity0
cn.viscosity, cn.boundary_viscosity = unit.from_real_kinematic_viscosity(
    parameterize_kinematic_viscosity(kinematic_viscosity_real))

# rigids
max_num_contacts = 512
pile = dp.Pile(dp, runner, max_num_contacts)

container_width_real = 0.24
container_width = unit.from_real_length(container_width_real)
container_dim = dp.f3(container_width, container_width, container_width)
container_mesh = al.Mesh()
container_mesh.set_box(container_dim, 8)
container_distance = dp.BoxDistance.create(container_dim, outset=0.46153312)
container_extent = container_distance.aabb_max - container_distance.aabb_min
container_res_float = container_extent / particle_radius
container_res = al.uint3(int(container_res_float.x),
                         int(container_res_float.y),
                         int(container_res_float.z))
print('container_res', container_res)
pile.add(
    container_distance,
    container_res,
    # al.uint3(60, 60, 60),
    sign=-1,
    collision_mesh=container_mesh,
    mass=0,
    restitution=0.8,
    friction=0.3)

piv_real_freq = 500.0
robot_real_freq = 200.0
piv_real_interval = 1.0 / piv_real_freq
interpolator = RigidInterpolator(
    dp, unit, '/media/kennychufk/vol1bk0/20210409_141511/Trace07.csv')
robot_sim_shift_real = dp.f3(0.52, container_width_real * 0.5 - 0.4479564, 0)
robot_sim_shift = unit.from_real_length(robot_sim_shift_real)
robot_time_offset = read_file_int(
    f'{args.truth_dir}/robot-time-offset') / piv_real_freq
pile.x[0] = interpolator.get_x(robot_time_offset) + robot_sim_shift

pile.reallocate_kinematics_on_device()
pile.set_gravity(gravity)

block_mode = 2
box_min = dp.f3(container_width * -0.5, 0, container_width * -0.5)
box_max = dp.f3(container_width * 0.5, container_width, container_width * 0.5)
fluid_block_capacity = dp.Runner.get_fluid_block_num_particles(
    mode=block_mode, box_min=box_min, box_max=box_max, particle_radius=spacing)
liquid_mass = unit.from_real_mass(
    read_file_float(f'{args.truth_dir}/mass.txt'))
num_particles = int(liquid_mass / particle_mass)
if (fluid_block_capacity < num_particles):
    print("fluid block is too small to hold the liquid mass")
    sys.exit(0)

container_aabb_range = container_distance.aabb_max - container_distance.aabb_min
container_aabb_range_per_h = container_aabb_range / kernel_radius
grid_res = al.uint3(int(np.ceil(container_aabb_range_per_h.x)),
                    int(np.ceil(container_aabb_range_per_h.y)),
                    int(np.ceil(container_aabb_range_per_h.z))) + 4
grid_offset = al.int3(
    -(int(grid_res.x) // 2) - 2,
    -int(np.ceil(container_distance.outset / kernel_radius)) - 1,
    -(int(grid_res.z) // 2) - 2)

cni.grid_res = grid_res
cni.grid_offset = grid_offset
cni.max_num_particles_per_cell = 64
cni.max_num_neighbors_per_particle = 64
margin_multiple = 3
cn.set_hcp_grid(
    cni, container_distance.aabb_min + pile.x[0] +
    unit.from_real_length(dp.f3(-0.05, -0.05, -0.05)) * margin_multiple,
    container_distance.aabb_max + pile.x[0] +
    unit.from_real_length(dp.f3(0.05, 0.05, 0.05)) * margin_multiple, spacing)

cn.density_ghost_threshold = cn.density0 * 0.94
solver = dp.SolverI(runner,
                    pile,
                    dp,
                    num_particles,
                    max_num_provisional_ghosts=num_particles * 10,
                    max_num_ghosts=num_particles * 5,
                    num_ushers=0,
                    enable_surface_tension=False,
                    enable_vorticity=False,
                    graphical=args.display)
if args.display:
    particle_normalized_attr = dp.create_graphical_like(
        solver.particle_density)
solver.relax_rate = 1.0
solver.num_ghost_relaxation = 40
solver.ghost_fluid_density_threshold = cn.density0 * 0.9
solver.num_particles = num_particles
solver.max_dt = unit.from_real_time(0.05 * unit.rl)
solver.initial_dt = solver.max_dt
solver.min_dt = 0
solver.cfl = 0.4
# solver.density_error_tolerance = 1e-4

dp.map_graphical_pointers()
runner.launch_create_fluid_block(solver.particle_x,
                                 num_particles,
                                 offset=0,
                                 particle_radius=spacing,
                                 mode=block_mode,
                                 box_min=box_min,
                                 box_max=box_max)
dp.unmap_graphical_pointers()

if args.display:
    display_proxy.set_camera(unit.from_real_length(al.float3(-0.6, 0.6, 0.0)),
                             unit.from_real_length(al.float3(0, 0.04, 0)))
    display_proxy.set_clip_planes(unit.to_real_length(1), container_width * 20)
    colormap_tex = display_proxy.create_colormap_viridis()

    display_proxy.add_particle_shading_program(solver.particle_x,
                                               particle_normalized_attr,
                                               colormap_tex,
                                               solver.particle_radius, solver)

rest_state_achieved = False
last_tranquillized = 0.0
rest_state_achieved = False
while not rest_state_achieved:
    dp.map_graphical_pointers()
    for substep_id in range(10):
        v_rms = np.sqrt(
            runner.sum(solver.particle_cfl_v2, solver.num_particles) /
            solver.num_particles)
        sufficient_time_after_tranquilizing = unit.to_real_time(
            solver.t - last_tranquillized) > 0.4
        if unit.to_real_time(solver.t - last_tranquillized) > 0.45:
            solver.max_dt = unit.from_real_time(0.18 * unit.rl)
            solver.particle_v.set_zero()
            solver.reset_solving_var()
            last_tranquillized = solver.t
        elif sufficient_time_after_tranquilizing and unit.to_real_velocity(
                v_rms) < 0.06:
            print("rest state achieved at", unit.to_real_time(solver.t))
            rest_state_achieved = True
        solver.step()
    if dp.has_display():
        solver.normalize(solver.particle_density, particle_normalized_attr,
                         cn.density0 * 0.5, cn.density0 * 1.1)
    dp.unmap_graphical_pointers()
    if dp.has_display():
        tmp_num_ghosts = solver.num_ghosts
        solver.num_ghosts = tmp_num_ghosts // 2
        display_proxy.draw()
        solver.num_ghosts = tmp_num_ghosts
    print(
        f"{int(sufficient_time_after_tranquilizing)} t = {unit.to_real_time(solver.t) } dt = {unit.to_real_time(solver.dt)} cfl = {solver.utilized_cfl} vrms={unit.to_real_velocity(v_rms)} max_v={unit.to_real_velocity(np.sqrt(solver.max_v2))} num solves = {solver.num_density_solve}"
    )

for dummy_itr in range(1):
    solver.reset_solving_var()
    solver.t = 0
    sample_x_piv = np.load(f'{args.truth_dir}/mat_results/pos.npy').reshape(
        -1, 2)
    sample_x_np = np.zeros((len(sample_x_piv), 3), dtype=dp.default_dtype)
    sample_x_np[:, 2] = sample_x_piv[:, 0]
    sample_x_np[:, 1] = sample_x_piv[:, 1]

    sampling = FluidSample(dp, unit.from_real_length(sample_x_np))
    ground_truth = dp.create_coated_like(sampling.sample_data3)

    truth_v_piv = np.load(
        f'{args.truth_dir}/mat_results/vel_original.npy').reshape(
            -1, len(sample_x_piv), 2)
    truth_v_np = np.zeros((*truth_v_piv.shape[:-1], 3))
    truth_v_np[..., 2] = truth_v_piv[..., 0]
    truth_v_np[..., 1] = truth_v_piv[..., 1]

    num_frames = len(truth_v_piv)

    visual_real_freq = 30.0
    visual_real_interval = 1.0 / visual_real_freq
    next_visual_frame_id = 0
    visual_x_scaled = dp.create_coated_like(solver.particle_x)

    timestamp_str, timestamp_hash = get_timestamp_and_hash()
    save_dir = Path(f'{Path(args.truth_dir).name}-{timestamp_hash}')
    save_dir.mkdir(parents=True)
    save_dir_visual = Path(save_dir, 'visual')
    save_dir_visual.mkdir(parents=True)
    save_dir_piv = Path(save_dir, 'piv')
    save_dir_piv.mkdir(parents=True)
    sim_v_np = np.zeros_like(truth_v_np)
    sim_errors = np.zeros(num_frames)
    for frame_id in range(num_frames):
        target_t = unit.from_real_time(frame_id * piv_real_interval)
        if dp.has_display():
            display_proxy.draw()

        dp.map_graphical_pointers()
        while (solver.t < target_t):
            solver.step()
            robot_offset_t = solver.t + robot_time_offset
            if solver.t >= unit.from_real_time(
                    next_visual_frame_id * visual_real_interval):
                visual_x_scaled.set_from(solver.particle_x)
                visual_x_scaled.scale(unit.to_real_length(1))
                visual_x_scaled.write_file(
                    f'{str(save_dir_visual)}/visual-x-{next_visual_frame_id}.alu',
                    solver.num_particles)
                pile.write_file(
                    f'{str(save_dir_visual)}/visual-{next_visual_frame_id}.pile',
                    unit.to_real_length(1), unit.to_real_velocity(1),
                    unit.to_real_angular_velocity(1))
                next_visual_frame_id += 1
            pile.x[0] = interpolator.get_x(robot_offset_t) + robot_sim_shift
            pile.v[0] = interpolator.get_v(robot_offset_t)
        sampling.prepare_neighbor_and_boundary(runner, solver)
        simulation_v_real = sampling.sample_velocity(runner, solver)
        simulation_v_real.scale(unit.to_real_velocity(1))
        sim_v_np[frame_id] = simulation_v_real.get()
        ground_truth.set(truth_v_np[frame_id])
        mse_yz = runner.calculate_mse_yz(simulation_v_real, ground_truth,
                                         sampling.num_samples)
        print(unit.to_real_time(target_t), unit.to_real_velocity_mse(mse_yz),
              pile.v[0])
        sim_errors[frame_id] = mse_yz

        if dp.has_display():
            # solver.normalize(solver.particle_v, particle_normalized_attr, 0,
            #                  unit.from_real_velocity(0.02))
            solver.normalize(solver.particle_density, particle_normalized_attr,
                             cn.density0 * 0.5, cn.density0 * 1.1)
        dp.unmap_graphical_pointers()

    np.save(f'{str(save_dir_piv)}/sim_v_real.npy', sim_v_np)
    np.save(f'{str(save_dir_piv)}/truth_v_real.npy', truth_v_np)
    np.save(f'{str(save_dir_piv)}/sim_errors.npy', sim_errors)

    dp.remove(ground_truth)
    dp.remove(visual_x_scaled)
    if args.display:
        dp.remove(particle_normalized_attr)
