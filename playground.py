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
import wandb
import torch

from rl import TD3, OrnsteinUhlenbeckProcess, GaussianNoise
from util import Unit, FluidSamplePellets, parameterize_kinematic_viscosity, get_state_dim, get_action_dim, make_state, set_usher_param, get_coil_x_from_com, BuoySpec

parser = argparse.ArgumentParser(description='RL playground')
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--cache-dir', type=str, default='.')
parser.add_argument('--truth-dir', type=str, required=True)
parser.add_argument('--display', metavar='d', type=bool, default=False)
parser.add_argument('--block-scan', metavar='s', type=bool, default=False)
parser.add_argument('--replay-buffer', type=str, default='')
parser.add_argument('--validate-model', type=str, default='')
args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

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

real_kernel_radius = 2**-6
density0_real = 1156.2
unit = Unit(real_kernel_radius=real_kernel_radius,
            real_density0=density0_real,
            real_gravity=-9.80665)

cn.set_kernel_radius(kernel_radius)
cn.set_particle_attr(particle_radius, particle_mass, density0)
cn.boundary_epsilon = 1e-9
cn.gravity = gravity

container_pellet_filename = '/home/kennychufk/workspace/pythonWs/alluvion-optim/cube24-2to-6.alu'

container_num_pellets = dp.get_alu_info(container_pellet_filename)[0][0]
num_pellets = container_num_pellets

# rigids
max_num_contacts = 512
pile = dp.Pile(dp, runner, max_num_contacts, al.VolumeMethod.pellets,
               num_pellets)

## ================== using cube
container_width = unit.from_real_length(0.24)
container_dim = dp.f3(container_width, container_width, container_width)
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
buoy_spec = BuoySpec(dp, unit)

pile.reallocate_kinematics_on_device()
pile.set_gravity(gravity)
cn.contact_tolerance = particle_radius

container_aabb_range_per_h = container_extent / kernel_radius
cni.grid_res = al.uint3(int(math.ceil(container_aabb_range_per_h.x)),
                        int(math.ceil(container_aabb_range_per_h.y)),
                        int(math.ceil(container_aabb_range_per_h.z))) + 4
cni.grid_offset = al.int3(
    int(container_distance.aabb_min.x) - 2,
    int(container_distance.aabb_min.y) - 2,
    int(container_distance.aabb_min.z) - 2)
cni.max_num_particles_per_cell = 64
cni.max_num_neighbors_per_particle = 64

max_num_buoys = 100
max_num_particles = 7368  # TODO: calculate
solver = dp.SolverI(runner,
                    pile,
                    dp,
                    max_num_particles,
                    num_ushers=max_num_buoys,
                    enable_surface_tension=False,
                    enable_vorticity=False,
                    graphical=args.display)

usher_sampling = FluidSamplePellets(
    dp, np.zeros((max_num_buoys, 3), dp.default_dtype))

solver.max_dt = unit.from_real_time(0.0005)
solver.initial_dt = solver.max_dt
solver.min_dt = 0
solver.cfl = 0.2

if args.display:
    display_proxy.set_camera(unit.from_real_length(al.float3(0, 0.06, 0.4)),
                             unit.from_real_length(al.float3(0, 0.0, 0)))
    display_proxy.set_clip_planes(unit.to_real_length(1),
                                  container_distance.aabb_max.z * 20)
    colormap_tex = display_proxy.create_colormap_viridis()
    particle_normalized_attr = dp.create_graphical((max_num_particles), 1)

    display_proxy.add_particle_shading_program(solver.particle_x,
                                               particle_normalized_attr,
                                               colormap_tex,
                                               solver.particle_radius, solver)
    display_proxy.add_pile_shading_program(pile)

next_force_time = 0.0
remaining_force_time = 0.0

score_history = deque(maxlen=100)
episode_id = 0

ground_truth_dir_list = [
    f"{args.truth_dir}/rltruth-be268318-0526.07.32.30/",
    f"{args.truth_dir}/rltruth-5caefe43-0526.14.46.12/",
    f"{args.truth_dir}/rltruth-e8edf09d-0526.18.34.19/",
    f"{args.truth_dir}/rltruth-6de1d91b-0526.09.31.47/",
    f"{args.truth_dir}/rltruth-3b860b54-0526.23.12.15/",
    f"{args.truth_dir}/rltruth-eb3494c1-0527.00.32.34/",
    f"{args.truth_dir}/rltruth-e9ba71d8-0527.01.52.31/"
]

# # 100 markers
# ground_truth_dir_list = [
#     f"{args.truth_dir}/rltruth-70eb294d-0528.12.35.00/",
#     f"{args.truth_dir}/rltruth-9b57dabc-0528.17.09.24/",
#     f"{args.truth_dir}/rltruth-1b06a408-0528.18.46.50/",
#     f"{args.truth_dir}/rltruth-ffc2674d-0528.20.27.56/",
#     f"{args.truth_dir}/rltruth-f218e02c-0528.22.05.13/",
#     f"{args.truth_dir}/rltruth-a4f6c703-0528.23.50.15/",
#     f"{args.truth_dir}/rltruth-63caecfb-0529.01.28.09/"
# ]

truth_real_freq = 100.0
truth_real_interval = 1.0 / truth_real_freq
next_truth_frame_id = 0
truth_buoy_pile_real = dp.Pile(dp, runner, 0)
for i in range(max_num_buoys):
    truth_buoy_pile_real.add(dp.SphereDistance.create(0), al.uint3(64, 64, 64))

max_xoffset = 0.05
max_voffset = 0.04
max_focal_dist = 0.20
min_usher_kernel_radius = 0.02
max_usher_kernel_radius = 0.06
max_strength = 720

agent = TD3(actor_lr=3e-4,
            critic_lr=3e-4,
            critic_weight_decay=0,
            state_dim=get_state_dim(),
            action_dim=get_action_dim(),
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
            learn_after=10000,
            replay_size=36000000,
            hidden_sizes=[2048, 2048, 1024],
            actor_final_scale=0.05 / np.sqrt(1000),
            critic_final_scale=1,
            soft_update_rate=0.005,
            batch_size=256)
if len(args.replay_buffer) > 0:
    agent.memory.load(args.replay_buffer)
validation_mode = (len(args.validate_model) > 0)
if validation_mode:
    agent.load_models(args.validate_model)
wandb.init(project='alluvion-rl')
config = wandb.config
config.actor_lr = agent.actor_lr
config.critic_lr = agent.critic_lr
config.critic_weight_decay = agent.critic_weight_decay
config.state_dim = agent.state_dim
config.action_dim = agent.action_dim
config.hidden_sizes = agent.hidden_sizes
config.max_action = agent.target_actor.max_action
config.soft_update_rate = agent.soft_update_rate
config.gamma = agent.gamma
config.replay_size = agent.replay_size
config.actor_final_scale = agent.actor_final_scale
config.critic_final_scale = agent.critic_final_scale
# config.sigma = agent.expl_noise_func.sigma
# config.theta = agent.expl_noise_func.theta
config.learn_after = agent.learn_after
config.seed = args.seed

wandb.watch(agent.critic)
if len(args.replay_buffer) > 0:
    agent.learn_after = 0

with open('switch', 'w') as f:
    f.write('1')
sample_step = 0
dir_id = 0
dumped_buffer = (len(args.replay_buffer) > 0)
while True:
    with open('switch', 'r') as f:
        if f.read(1) == '0':
            break
    # dir_id = 0 if args.block_scan else random.randrange(
    #     len(ground_truth_dir_list))
    dir_id = (dir_id + 1) % len(ground_truth_dir_list)
    ground_truth_dir = ground_truth_dir_list[dir_id]
    print(dir_id, ground_truth_dir)

    unit = Unit(
        real_kernel_radius=real_kernel_radius,
        real_density0=np.load(f'{ground_truth_dir}/density0_real.npy').item(),
        real_gravity=-9.80665)
    kinematic_viscosity_real = np.load(
        f'{ground_truth_dir}/kinematic_viscosity_real.npy').item()

    cn.set_kernel_radius(kernel_radius)
    cn.set_particle_attr(particle_radius, particle_mass, density0)
    cn.boundary_epsilon = 1e-9
    cn.gravity = gravity
    cn.viscosity, cn.boundary_viscosity = unit.from_real_kinematic_viscosity(
        parameterize_kinematic_viscosity(kinematic_viscosity_real)
    )  # TODO: change to parameterize_kinematic_viscosity_with_pellets

    fluid_mass = unit.from_real_mass(
        np.load(f'{ground_truth_dir}/fluid_mass.npy').item())
    num_particles = int(fluid_mass / cn.particle_mass)
    solver.num_particles = num_particles

    num_buoys = dp.Pile.get_size_from_file(f'{ground_truth_dir}/0.pile') - 2
    solver.usher.reset()
    solver.usher.num_ushers = num_buoys
    print('num_particles', num_particles, 'num_buoys', num_buoys)

    initial_particle_x_filename = f'{args.cache_dir}/playground_bead_x-2to-6.alu'
    initial_particle_v_filename = f'{args.cache_dir}/playground_bead_v-2to-6.alu'
    initial_particle_pressure_filename = f'{args.cache_dir}/playground_bead_p-2to-6.alu'
    if not Path(initial_particle_x_filename).is_file():
        fluid_block_mode = 0
        dp.map_graphical_pointers()
        runner.launch_create_fluid_block(solver.particle_x,
                                         solver.num_particles,
                                         offset=0,
                                         particle_radius=particle_radius,
                                         mode=fluid_block_mode,
                                         box_min=container_distance.aabb_min,
                                         box_max=container_distance.aabb_max)
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
    sampling = FluidSamplePellets(dp, f'{ground_truth_dir}/sample-x.alu')
    mask = dp.create_coated((sampling.num_samples), 1, np.uint32)
    sampling.sample_x.scale(unit.from_real_length(1))
    ground_truth = dp.create_coated_like(sampling.sample_data3)
    sum_v2 = unit.from_real_velocity_mse(
        np.load(f'{ground_truth_dir}/sum_v2.npy').item())
    max_v2 = unit.from_real_velocity_mse(
        np.load(f'{ground_truth_dir}/max_v2.npy').item())
    dp.map_graphical_pointers()
    solver.update_particle_neighbors()
    truth_buoy_pile_real.read_file(f'{ground_truth_dir}/0.pile',
                                   num_rigids=num_buoys,
                                   dst_offset=0,
                                   src_offset=1)
    coil_x_real = get_coil_x_from_com(dp, unit, buoy_spec,
                                      truth_buoy_pile_real, num_buoys)
    # set positions for sampling around buoys in simulation
    coil_x_np = unit.from_real_length(coil_x_real)
    usher_sampling.sample_x.set(coil_x_np)
    usher_sampling.prepare_neighbor_and_boundary(runner, solver)
    usher_sampling.sample_density(runner)
    usher_sampling.sample_velocity(runner, solver)
    # usher_sampling.sample_vorticity(runner, solver)

    state_aggregated = make_state(dp, unit, kinematic_viscosity_real,
                                  truth_buoy_pile_real, coil_x_real,
                                  usher_sampling, num_buoys)

    dp.unmap_graphical_pointers()

    score = 0
    error_sum = 0

    num_frames = 1000
    truth_real_freq = 100.0
    truth_real_interval = 1.0 / truth_real_freq

    visual_real_freq = 30.0
    visual_real_interval = 1.0 / visual_real_freq
    next_visual_frame_id = 0
    visual_x_scaled = dp.create_coated_like(solver.particle_x)

    num_frames_in_scenario = 0
    for frame_id in range(num_frames - 1):
        num_frames_in_scenario += 1
        target_t = unit.from_real_time(frame_id * truth_real_interval)

        truth_buoy_pile_real.read_file(f'{ground_truth_dir}/{frame_id}.pile',
                                       num_rigids=num_buoys,
                                       dst_offset=0,
                                       src_offset=1)
        coil_x_real = get_coil_x_from_com(dp, unit, buoy_spec,
                                          truth_buoy_pile_real, num_buoys)
        coil_x_np = unit.from_real_length(coil_x_real)
        usher_sampling.sample_x.set(coil_x_np)

        if sample_step < agent.learn_after:
            action_aggregated = np.zeros((num_buoys, agent.action_dim))
            for buoy_id in range(num_buoys):
                action_aggregated[buoy_id] = agent.uniform_random_action()
        else:
            action_aggregated = agent.get_action(
                state_aggregated, enable_noise=not validation_mode)
        if np.sum(np.isnan(action_aggregated)) > 0:
            print(state_aggregated, action_aggregated)
            sys.exit(0)
        action_aggregated_converted = agent.actor.from_normalized_action(
            action_aggregated)
        set_usher_param(solver.usher, dp, unit, truth_buoy_pile_real,
                        coil_x_real, action_aggregated_converted, num_buoys)
        dp.map_graphical_pointers()
        while (solver.t < target_t):
            solver.step()
            if validation_mode and solver.t >= unit.from_real_time(
                    next_visual_frame_id * visual_real_interval):
                visual_x_scaled.set_from(solver.particle_x)
                visual_x_scaled.scale(unit.to_real_length(1))
                visual_x_scaled.write_file(
                    f'val/visual-x-{next_visual_frame_id}.alu',
                    solver.num_particles)
                pile.write_file(f'val/visual-{next_visual_frame_id}.pile',
                                unit.to_real_length(1),
                                unit.to_real_velocity(1),
                                unit.to_real_angular_velocity(1))
                next_visual_frame_id += 1
        truth_buoy_pile_real.read_file(f'{ground_truth_dir}/{frame_id+1}.pile',
                                       num_rigids=num_buoys,
                                       dst_offset=0,
                                       src_offset=1)
        coil_x_real = get_coil_x_from_com(dp, unit, buoy_spec,
                                          truth_buoy_pile_real, num_buoys)
        coil_x_np = unit.from_real_length(coil_x_real)
        usher_sampling.sample_x.set(
            coil_x_np)  # NOTE: should set to new sampling points?
        usher_sampling.prepare_neighbor_and_boundary(runner, solver)
        usher_sampling.sample_density(runner)
        usher_sampling.sample_velocity(runner, solver)
        # usher_sampling.sample_vorticity(runner, solver)
        new_state_aggregated = make_state(dp, unit, kinematic_viscosity_real,
                                          truth_buoy_pile_real, coil_x_real,
                                          usher_sampling, num_buoys)

        # find reward
        sampling.prepare_neighbor_and_boundary(runner, solver)
        simulation_v_real = sampling.sample_velocity(runner, solver)
        simulation_v_real.scale(unit.to_real_velocity(1))

        ground_truth.read_file(f'{ground_truth_dir}/v-{frame_id+1}.alu')
        mask.read_file(f'{ground_truth_dir}/mask-{frame_id+1}.alu')
        reconstruction_error = runner.calculate_mse_masked(
            simulation_v_real, ground_truth, mask, sampling.num_samples)
        reward = -reconstruction_error / max_v2
        error_sum += reconstruction_error
        early_termination = False
        grid_anomaly = dp.coat(solver.grid_anomaly).get()[0]
        if reward < -3 or grid_anomaly > 0:
            reward -= 3
            print(f'early termination {reward}')
            early_termination = True

        if not validation_mode:
            for buoy_id in range(num_buoys):
                agent.remember(
                    state_aggregated[buoy_id], action_aggregated[buoy_id],
                    reward, new_state_aggregated[buoy_id],
                    int(early_termination or frame_id == (num_frames - 2)))
            if sample_step >= agent.learn_after:  # as memory size is num_buoys * sample_step
                if not dumped_buffer:
                    # agent.memory.save('buffer-dump')
                    dumped_buffer = True
                agent.learn()
        else:
            # NOTE: Saving all required data for animation. Only for validating a single scenario.
            np.save(f'val/action-{frame_id}.npy', action_aggregated_converted)
            np.save(f'val/state-{frame_id}.npy', state_aggregated)
            buoys_q0, buoys_q1 = agent.get_value(state_aggregated,
                                                 action_aggregated)
            np.save(f'val/value0-{frame_id}.npy',
                    buoys_q0.cpu().detach().numpy())
            np.save(f'val/value1-{frame_id}.npy',
                    buoys_q1.cpu().detach().numpy())
            simulation_v_real.write_file(f'val/v-{frame_id}.alu',
                                         sampling.num_samples)
            np.save(f'val/reward-{frame_id}.npy', reward)

        sample_step += 1
        state_aggregated = new_state_aggregated

        if dp.has_display():
            solver.normalize(solver.particle_v, particle_normalized_attr, 0,
                             unit.from_real_velocity(0.02))
        dp.unmap_graphical_pointers()
        if dp.has_display():
            display_proxy.draw()
        if early_termination:
            break
    score = -error_sum / sum_v2
    score_history.append(score)

    if not validation_mode and episode_id % 50 == 0:
        save_dir = f"artifacts/{wandb.run.id}/models/{episode_id}"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        agent.save_models(save_dir)
    log_object = {'score': score}
    if len(score_history) == score_history.maxlen:
        log_object['score100'] = np.mean(list(score_history))
    wandb.log(log_object)
    if validation_mode:
        np.save('val/score.npy', score)
        print('score', score)
    dp.remove(mask)
    dp.remove(ground_truth)
    dp.remove(visual_x_scaled)
    sampling.destroy_variables()
    episode_id += 1
    if args.block_scan:
        break
    if validation_mode:
        break
