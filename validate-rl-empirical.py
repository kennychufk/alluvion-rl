import sys
import argparse
import math
import random
import os
from collections import deque

import alluvion as al
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error
import torch

from ddpg_torch import TD3, GaussianNoise
from util import Unit, FluidSample, parameterize_kinematic_viscosity, BuoyInterpolator, get_obs_dim, get_act_dim, make_obs, set_usher_param
from util import get_timestamp_and_hash
from util import read_file_int, read_file_float


def set_pile_from_interpolators(truth_buoy_pile_real, buoy_interpolators,
                                t_real):
    for buoy_id in range(truth_buoy_pile_real.get_size()):
        truth_buoy_pile_real.x[buoy_id] = buoy_interpolators[buoy_id].get_x(
            t_real)
        truth_buoy_pile_real.q[buoy_id] = buoy_interpolators[buoy_id].get_q(
            t_real)
        truth_buoy_pile_real.v[buoy_id] = buoy_interpolators[buoy_id].get_v(
            t_real)


parser = argparse.ArgumentParser(
    description='RL validation with empirical data')
parser.add_argument('--truth-dir', type=str, required=True)
parser.add_argument('--cache-dir', type=str, default='.')
parser.add_argument('--display', metavar='d', type=bool, default=False)
parser.add_argument('--validate-model', type=str, required=True)
args = parser.parse_args()

dp = al.Depot(np.float32)
cn = dp.cn
cni = dp.cni

if args.display:
    dp.create_display(800, 600, "", False)

display_proxy = dp.get_display_proxy() if args.display else None
runner = dp.Runner()

particle_radius = 0.25
kernel_radius = 1.0
density0 = 1.0
cubical_particle_volume = 8 * particle_radius * particle_radius * particle_radius
volume_relative_to_cube = 0.8
particle_mass = cubical_particle_volume * volume_relative_to_cube * density0

gravity = dp.f3(0, -1, 0)

real_kernel_radius = 0.020  # TODO: try changing
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

## ================== using cube
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
## ================== using cube

pile.reallocate_kinematics_on_device()
pile.set_gravity(gravity)
cn.contact_tolerance = particle_radius

container_aabb_range_per_h = container_extent / kernel_radius
cni.grid_res = al.uint3(int(math.ceil(container_aabb_range_per_h.x)),
                        int(math.ceil(container_aabb_range_per_h.y)),
                        int(math.ceil(container_aabb_range_per_h.z))) + 4
cni.grid_offset = al.int3(
    -(int(cni.grid_res.x) // 2) - 2,
    -int(np.ceil(container_distance.outset / kernel_radius)) - 1,
    -(int(cni.grid_res.z) // 2) - 2)
cni.max_num_particles_per_cell = 64
cni.max_num_neighbors_per_particle = 64

used_buoy_ids = np.load(f'{args.truth_dir}/rec/marker_ids.npy')
num_buoys = len(used_buoy_ids)

block_mode = 0
box_min = dp.f3(container_width * -0.5, 0, container_width * -0.5)
box_max = dp.f3(container_width * 0.5, container_width, container_width * 0.5)
fluid_block_capacity = dp.Runner.get_fluid_block_num_particles(
    mode=block_mode,
    box_min=box_min,
    box_max=box_max,
    particle_radius=particle_radius)
liquid_mass = unit.from_real_mass(
    read_file_float(f'{args.truth_dir}/mass.txt'))
num_particles = int(liquid_mass / particle_mass)
if (fluid_block_capacity < num_particles):
    print("fluid block is too small to hold the liquid mass")
    sys.exit(0)

solver = dp.SolverI(runner,
                    pile,
                    dp,
                    num_particles,
                    num_ushers=num_buoys,
                    enable_surface_tension=False,
                    enable_vorticity=False,
                    graphical=args.display)

usher_sampling = FluidSample(dp, np.zeros((num_buoys, 3), dp.default_dtype))
solver.num_particles = num_particles
solver.max_dt = unit.from_real_time(0.05 * unit.rl)
solver.initial_dt = solver.max_dt
solver.min_dt = 0
solver.cfl = 0.4

if args.display:
    display_proxy.set_camera(unit.from_real_length(al.float3(0, 0.06, 0.4)),
                             unit.from_real_length(al.float3(0, 0.0, 0)))
    display_proxy.set_clip_planes(unit.to_real_length(1),
                                  container_distance.aabb_max.z * 20)
    colormap_tex = display_proxy.create_colormap_viridis()
    particle_normalized_attr = dp.create_graphical((num_particles), 1)

    display_proxy.add_particle_shading_program(solver.particle_x,
                                               particle_normalized_attr,
                                               colormap_tex,
                                               solver.particle_radius, solver)
    display_proxy.add_pile_shading_program(pile)

next_force_time = 0.0
remaining_force_time = 0.0

im_real_freq = 100.0
im_real_interval = 1.0 / im_real_freq
next_truth_frame_id = 0
truth_buoy_pile_real = dp.Pile(dp, runner, 0)
for i in range(num_buoys):
    truth_buoy_pile_real.add(dp.SphereDistance.create(0), al.uint3(64, 64, 64))

piv_real_freq = 500.0
piv_real_interval = 1.0 / piv_real_freq

max_xoffset = 0.05
max_voffset = 0.04
max_focal_dist = 0.20
min_usher_kernel_radius = 0.02
max_usher_kernel_radius = 0.06
max_strength = 720

pile.x[0] = dp.f3(0, container_width * 0.5, 0)

agent = TD3(actor_lr=3e-4,
            critic_lr=3e-4,
            critic_weight_decay=0,
            obs_dim=get_obs_dim(),
            act_dim=get_act_dim(),
            expl_noise_func=GaussianNoise(std_dev=0.1),
            gamma=0.95,
            min_action=np.array([
                -max_xoffset, -max_xoffset, -max_xoffset, -max_xoffset,
                -max_xoffset, -max_xoffset, -max_xoffset, -max_xoffset,
                -max_xoffset, -max_voffset, -max_voffset, -max_voffset,
                -max_voffset, -max_voffset, -max_voffset, -max_voffset,
                -max_voffset, -max_voffset, 0.0, min_usher_kernel_radius, 0
            ]),
            max_action=np.array([
                +max_xoffset, +max_xoffset, +max_xoffset, +max_xoffset,
                +max_xoffset, +max_xoffset, +max_xoffset, +max_xoffset,
                +max_xoffset, +max_voffset, +max_voffset, +max_voffset,
                +max_voffset, +max_voffset, +max_voffset, +max_voffset,
                +max_voffset, +max_voffset, max_focal_dist,
                max_usher_kernel_radius, max_strength
            ]),
            learn_after=0,
            replay_size=0,
            hidden_sizes=[2048, 2048, 1024],
            actor_final_scale=1,
            critic_final_scale=1,
            soft_update_rate=0.005,
            batch_size=256)

agent.load_models(args.validate_model)

for dummy_itr in range(1):
    solver.num_particles = num_particles
    print('num_particles', num_particles)

    initial_particle_x_filename = f'{args.cache_dir}/x{num_particles}.alu'
    initial_particle_v_filename = f'{args.cache_dir}/v{num_particles}.alu'
    initial_particle_pressure_filename = f'{args.cache_dir}/pressure{num_particles}.alu'
    if True:
        # if not Path(initial_particle_x_filename).is_file():
        dp.map_graphical_pointers()
        runner.launch_create_fluid_block(solver.particle_x,
                                         solver.num_particles,
                                         offset=0,
                                         particle_radius=particle_radius,
                                         mode=block_mode,
                                         box_min=box_min,
                                         box_max=box_max)
        dp.unmap_graphical_pointers()
        solver.t = 0
        last_tranquillized = 0.0
        rest_state_achieved = False
        while not rest_state_achieved:
            dp.map_graphical_pointers()
            for frame_interstep in range(10):
                v_rms = np.sqrt(
                    runner.sum(solver.particle_cfl_v2, solver.num_particles) /
                    solver.num_particles)
                if unit.to_real_time(solver.t - last_tranquillized) > 0.45:
                    solver.particle_v.set_zero()
                    solver.reset_solving_var()
                    last_tranquillized = solver.t
                elif unit.to_real_time(solver.t - last_tranquillized
                                       ) > 0.4 and unit.to_real_velocity(
                                           v_rms) < 0.01:
                    print("rest state achieved at",
                          unit.to_real_time(solver.t))
                    rest_state_achieved = True
                solver.step()
            dp.unmap_graphical_pointers()
            if dp.has_display():
                display_proxy.draw()
            print(
                f"t = {unit.to_real_time(solver.t) } dt = {unit.to_real_time(solver.dt)} cfl = {solver.utilized_cfl} vrms={unit.to_real_velocity(v_rms)} max_v={unit.to_real_velocity(np.sqrt(solver.max_v2))} num solves = {solver.num_density_solve}"
            )
        dp.map_graphical_pointers()
        solver.particle_x.write_file(initial_particle_x_filename,
                                     solver.num_particles)
        solver.particle_v.write_file(initial_particle_v_filename,
                                     solver.num_particles)
        solver.particle_pressure.write_file(initial_particle_pressure_filename,
                                            solver.num_particles)
        dp.unmap_graphical_pointers()
    else:
        dp.map_graphical_pointers()
        solver.particle_x.read_file(initial_particle_x_filename)
        solver.particle_v.read_file(initial_particle_v_filename)
        solver.particle_pressure.read_file(initial_particle_pressure_filename)
        dp.unmap_graphical_pointers()

    solver.reset_solving_var()
    solver.t = 0
    sample_x_piv = np.load(f'{args.truth_dir}/mat_results/pos.npy').reshape(
        -1, 2)
    sample_x_np = np.zeros((len(sample_x_piv), 3), dtype=dp.default_dtype)
    sample_x_np[:, 2] = sample_x_piv[:, 0]
    sample_x_np[:, 1] = sample_x_piv[:, 1]

    sampling = FluidSample(dp, unit.from_real_length(sample_x_np))
    ground_truth = dp.create_coated_like(sampling.sample_data3)
    mask = dp.create_coated((sampling.num_samples), 1, np.uint32)

    truth_v_piv = np.load(
        f'{args.truth_dir}/mat_results/vel_original.npy').reshape(
            -1, len(sample_x_piv), 2)
    truth_v_np = np.zeros((*truth_v_piv.shape[:-1], 3))
    truth_v_np[..., 2] = truth_v_piv[..., 0]
    truth_v_np[..., 1] = truth_v_piv[..., 1]

    mask_np = np.load(f'{args.truth_dir}/mat_results/mask.npy').reshape(
        -1, len(sample_x_piv))
    sum_v2 = 0

    filter_postfix = '-f20'
    buoy_trajectories = [
        np.load(
            f'{args.truth_dir}/rec/marker-{used_buoy_id}{filter_postfix}.npy')
        for used_buoy_id in used_buoy_ids
    ]
    buoy_interpolators = [
        BuoyInterpolator(dp, im_real_interval, trajectory)
        for trajectory in buoy_trajectories
    ]

    error_sum = 0

    num_frames = len(truth_v_piv)

    visual_real_freq = 30.0
    visual_real_interval = 1.0 / visual_real_freq
    next_visual_frame_id = 0
    visual_x_scaled = dp.create_coated_like(solver.particle_x)

    timestamp_str, timestamp_hash = get_timestamp_and_hash()
    save_dir = Path(f'recon-{Path(args.truth_dir).name}-{timestamp_hash}')
    save_dir.mkdir(parents=True)
    save_dir_visual = Path(save_dir, 'visual')
    save_dir_visual.mkdir(parents=True)
    save_dir_piv = Path(save_dir, 'piv')
    save_dir_piv.mkdir(parents=True)
    sim_v_np = np.zeros_like(truth_v_np)
    sim_errors = np.zeros(num_frames)
    zero3 = dp.create_coated_like(sampling.sample_data3)
    zero3.set_zero()
    for frame_id in range(num_frames - 1):
        target_t = unit.from_real_time(frame_id * piv_real_interval)

        im_frame_id = int(frame_id * im_real_freq / piv_real_freq)
        set_pile_from_interpolators(truth_buoy_pile_real, buoy_interpolators,
                                    unit.to_real_time(solver.t))
        # set positions for sampling around buoys in simulation
        coil_x_real = dp.coat(truth_buoy_pile_real.x).get()
        coil_x_np = unit.from_real_length(coil_x_real)
        usher_sampling.sample_x.set(coil_x_np)

        dp.map_graphical_pointers()
        usher_sampling.prepare_neighbor_and_boundary(runner, solver)
        usher_sampling.sample_density(runner)
        usher_sampling.sample_velocity(runner, solver)
        usher_sampling.sample_vorticity(runner, solver)

        obs_aggregated = make_obs(dp, unit, kinematic_viscosity_real,
                                  truth_buoy_pile_real, coil_x_real,
                                  usher_sampling, num_buoys)

        act_aggregated = agent.get_action(obs_aggregated, enable_noise=False)
        act_aggregated_converted = agent.actor.from_normalized_action(
            act_aggregated)
        set_usher_param(solver.usher, dp, unit, truth_buoy_pile_real,
                        coil_x_real, act_aggregated_converted, num_buoys)
        while (solver.t < target_t):
            solver.step()
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

        sampling.prepare_neighbor_and_boundary(runner, solver)
        simulation_v_real = sampling.sample_velocity(runner, solver)
        simulation_v_real.scale(unit.to_real_velocity(1))
        sim_v_np[frame_id] = simulation_v_real.get()
        ground_truth.set(truth_v_np[frame_id])
        mask.set(mask_np[frame_id])
        mse_yz = runner.calculate_mse_yz_masked(simulation_v_real,
                                                ground_truth, mask,
                                                sampling.num_samples)
        truth_yz_mean_norm_sqr = runner.calculate_mse_yz_masked(
            zero3, ground_truth, mask, sampling.num_samples)
        error_sum += mse_yz
        sum_v2 += truth_yz_mean_norm_sqr
        print(unit.to_real_time(target_t), unit.to_real_velocity_mse(mse_yz),
              pile.v[0])
        sim_errors[frame_id] = mse_yz

        if dp.has_display():
            solver.normalize(solver.particle_v, particle_normalized_attr, 0,
                             unit.from_real_velocity(0.02))
        dp.unmap_graphical_pointers()
        if dp.has_display():
            display_proxy.draw()

    score = -error_sum / sum_v2
    np.save(f'{str(save_dir_piv)}/score.npy', score)
    np.save(f'{str(save_dir_piv)}/sim_v_real.npy', sim_v_np)
    np.save(f'{str(save_dir_piv)}/truth_v_real.npy', truth_v_np)
    np.save(f'{str(save_dir_piv)}/sim_errors.npy', sim_errors)

    dp.remove(ground_truth)
    dp.remove(visual_x_scaled)
    dp.remove(zero3)
    if args.display:
        dp.remove(particle_normalized_attr)
    sampling.destroy_variables()
