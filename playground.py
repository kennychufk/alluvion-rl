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

from ddpg_torch import DDPGAgent, OrnsteinUhlenbeckProcess, TD3, GaussianNoise
from util import Unit, FluidSample, parameterize_kinematic_viscosity, get_quat3

parser = argparse.ArgumentParser(description='RL playground')
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--cache-dir', type=str, default='.')
parser.add_argument('--truth-dir', type=str, required=True)
parser.add_argument('--display', metavar='d', type=bool, default=False)
parser.add_argument('--block-scan', metavar='d', type=bool, default=False)
args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)


# using real unit except density: which is relative to density0
def make_obs(dp, unit, kinematic_viscosity_real, truth_buoy_pile_real,
             usher_sampling, num_buoys):
    obs_aggregated = np.zeros([num_buoys, agent.obs_dim], dp.default_dtype)
    buoy_x_real = dp.coat(truth_buoy_pile_real.x).get()
    buoy_v_real = dp.coat(truth_buoy_pile_real.v).get()
    buoy_q = dp.coat(truth_buoy_pile_real.q).get()
    buoy_q3 = get_quat3(buoy_q)
    buoy_omega = dp.coat(truth_buoy_pile_real.omega).get()

    sample_v_real = unit.to_real_velocity(usher_sampling.sample_data3.get())
    sample_density_relative = usher_sampling.sample_data1.get()
    sample_vort_real = unit.to_real_angular_velocity(
        usher_sampling.sample_vort.get())
    sample_container_kernel_sim = usher_sampling.sample_boundary_kernel.get(
    )[0]
    sample_container_kernel_vol_grad_real = unit.to_real_per_length(
        sample_container_kernel_sim[:, :3])
    sample_container_kernel_vol = sample_container_kernel_sim[:,
                                                              3]  # dimensionless

    for buoy_id in range(num_buoys):
        xi = buoy_x_real[buoy_id]
        vi = buoy_v_real[buoy_id]
        xij = xi - buoy_x_real
        d2 = np.sum(xij * xij, axis=1)
        dist_sort_index = np.argsort(d2)[1:]

        obs_aggregated[buoy_id] = np.concatenate(
            (xi, buoy_v_real[buoy_id], buoy_q3[buoy_id], buoy_omega[buoy_id],
             xij[dist_sort_index[0]], vi - buoy_v_real[dist_sort_index[0]],
             xij[dist_sort_index[1]], vi - buoy_v_real[dist_sort_index[1]],
             sample_v_real[buoy_id].flatten(),
             sample_density_relative[buoy_id],
             sample_vort_real[buoy_id].flatten(),
             sample_container_kernel_vol_grad_real[buoy_id].flatten(),
             sample_container_kernel_vol[buoy_id].flatten(), unit.rdensity0 /
             1000, kinematic_viscosity_real * 1e6, num_buoys / 9),
            axis=None)
    return obs_aggregated


def set_usher_param(usher, dp, unit, truth_buoy_pile_real, act_aggregated,
                    num_buoys):
    # [0:3] [3:6] [6:9] displacement from buoy x
    # [9:12] [12:15] [15:18] velocity offset from buoy v
    # [18] focal dist
    # [19] usher kernel radius
    # [20] strength
    xoffset_real = act_aggregated[:, 0:9].reshape(3, num_buoys, 3)
    voffset_real = act_aggregated[:, 9:18].reshape(3, num_buoys, 3)
    buoy_x_real = dp.coat(truth_buoy_pile_real.x).get()
    buoy_v_real = dp.coat(truth_buoy_pile_real.v).get()
    focal_x_real = np.zeros_like(xoffset_real)
    focal_v_real = np.zeros_like(voffset_real)
    for buoy_id in range(num_buoys):
        focal_x_real[:,
                     buoy_id] = xoffset_real[:, buoy_id] + buoy_x_real[buoy_id]
        focal_v_real[:,
                     buoy_id] = voffset_real[:, buoy_id] + buoy_v_real[buoy_id]

    dp.coat(usher.focal_x).set(unit.from_real_length(focal_x_real))
    dp.coat(usher.focal_v).set(unit.from_real_velocity(focal_v_real))

    dp.coat(usher.focal_dist).set(unit.from_real_length(act_aggregated[:, 18]))

    dp.coat(usher.usher_kernel_radius).set(
        unit.from_real_length(act_aggregated[:, 19]))

    dp.coat(usher.drive_strength).set(
        unit.from_real_angular_velocity(act_aggregated[:, 20]))


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

real_kernel_radius = 0.020
unit = Unit(real_kernel_radius=real_kernel_radius,
            real_density0=1000,
            real_gravity=-9.80665)

cn.set_kernel_radius(kernel_radius)
cn.set_particle_attr(particle_radius, particle_mass, density0)
cn.boundary_epsilon = 1e-9
cn.gravity = gravity

# rigids
max_num_contacts = 512
pile = dp.Pile(dp, runner, max_num_contacts)

## ================== using cube
container_scale = 1.0
container_option = ''
container_width = unit.from_real_length(0.24)
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
    int(container_distance.aabb_min.x) - 2,
    int(container_distance.aabb_min.y) - 2,
    int(container_distance.aabb_min.z) - 2)
cni.max_num_particles_per_cell = 64
cni.max_num_neighbors_per_particle = 64

max_num_buoys = 9
max_num_particles = 50000  # TODO: calculate
solver = dp.SolverI(runner,
                    pile,
                    dp,
                    max_num_particles,
                    num_ushers=max_num_buoys,
                    enable_surface_tension=False,
                    enable_vorticity=False,
                    graphical=args.display)

usher_sampling = FluidSample(dp, np.zeros((max_num_buoys, 3),
                                          dp.default_dtype))

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
    f.path for f in os.scandir(args.truth_dir) if f.is_dir()
]

truth_real_freq = 100.0
truth_real_interval = 1.0 / truth_real_freq
next_truth_frame_id = 0
truth_buoy_pile_real = dp.Pile(dp, runner, 0)
for i in range(max_num_buoys):
    truth_buoy_pile_real.add(dp.SphereDistance.create(0), al.uint3(64, 64, 64))

max_xoffset = 0.08
max_voffset = 0.06
max_focal_dist = 0.12
min_usher_kernel_radius = 0.02
max_usher_kernel_radius = 0.06
max_strength = 4000

agent = TD3(actor_lr=3e-4,
            critic_lr=3e-4,
            critic_weight_decay=0,
            obs_dim=38,
            act_dim=21,
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
            learn_after=1000000,
            hidden_sizes=[2048, 2048, 1024],
            actor_final_scale=1,
            critic_final_scale=1,
            soft_update_rate=0.005,
            batch_size=256)
wandb.init(project='alluvion-rl')
config = wandb.config
config.actor_lr = agent.actor_lr
config.critic_lr = agent.critic_lr
config.critic_weight_decay = agent.critic_weight_decay
config.obs_dim = agent.obs_dim
config.act_dim = agent.act_dim
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

with open('switch', 'w') as f:
    f.write('1')
sample_step = 0
while True:
    with open('switch', 'r') as f:
        if f.read(1) == '0':
            break
    dir_id = 0 if args.block_scan else random.randrange(
        len(ground_truth_dir_list))
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
        parameterize_kinematic_viscosity(kinematic_viscosity_real))

    fluid_mass = unit.from_real_mass(
        np.load(f'{ground_truth_dir}/fluid_mass.npy').item())
    num_particles = int(fluid_mass / cn.particle_mass)
    solver.num_particles = num_particles
    print('num_particles', num_particles)

    initial_particle_x_filename = f'{args.cache_dir}/x{num_particles}.alu'
    initial_particle_v_filename = f'{args.cache_dir}/v{num_particles}.alu'
    initial_particle_pressure_filename = f'{args.cache_dir}/pressure{num_particles}.alu'
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
    sampling = FluidSample(dp, f'{ground_truth_dir}/sample-x.alu')
    sampling.sample_x.scale(unit.from_real_length(1))
    ground_truth = dp.create_coated_like(sampling.sample_data3)
    dp.map_graphical_pointers()
    solver.update_particle_neighbors()
    num_buoys = dp.Pile.get_size_from_file(f'{ground_truth_dir}/0.pile') - 2
    truth_buoy_pile_real.read_file(f'{ground_truth_dir}/0.pile', num_buoys, 0,
                                   1)
    # set positions for sampling around buoys in simulation
    truth_buoy_x_np = unit.from_real_length(
        dp.coat(truth_buoy_pile_real.x).get())
    usher_sampling.sample_x.set(truth_buoy_x_np)
    usher_sampling.prepare_neighbor_and_boundary(runner, solver)
    usher_sampling.sample_density(runner)
    usher_sampling.sample_velocity(runner, solver)
    usher_sampling.sample_vorticity(runner, solver)

    obs_aggregated = make_obs(dp, unit, kinematic_viscosity_real,
                              truth_buoy_pile_real, usher_sampling, num_buoys)

    dp.unmap_graphical_pointers()

    score = 0

    num_frames = 1000
    truth_real_freq = 100.0
    truth_real_interval = 1.0 / truth_real_freq
    num_frames_in_scenario = 0
    for frame_id in range(num_frames - 1):
        num_frames_in_scenario += 1
        target_t = unit.from_real_time(frame_id * truth_real_interval)

        truth_buoy_pile_real.read_file(f'{ground_truth_dir}/{frame_id}.pile',
                                       num_buoys, 0, 1)
        truth_buoy_x_np = unit.from_real_length(
            dp.coat(truth_buoy_pile_real.x).get())
        usher_sampling.sample_x.set(truth_buoy_x_np)

        if sample_step < agent.learn_after:
            act_aggregated = np.zeros((num_buoys, agent.act_dim))
            for buoy_id in range(num_buoys):
                act_aggregated[buoy_id] = agent.uniform_random_action()
        else:
            act_aggregated = agent.get_action(obs_aggregated)
        if np.sum(np.isnan(act_aggregated)) > 0:
            print(obs_aggregated, act_aggregated)
            sys.exit(0)
        set_usher_param(solver.usher, dp, unit, truth_buoy_pile_real,
                        agent.actor.from_normalized_action(act_aggregated),
                        num_buoys)
        dp.map_graphical_pointers()
        while (solver.t < target_t):
            solver.step()
        truth_buoy_pile_real.read_file(f'{ground_truth_dir}/{frame_id+1}.pile',
                                       num_buoys, 0, 1)
        usher_sampling.prepare_neighbor_and_boundary(runner, solver)
        usher_sampling.sample_density(runner)
        usher_sampling.sample_velocity(runner, solver)
        usher_sampling.sample_vorticity(runner, solver)
        new_obs_aggregated = make_obs(dp, unit, kinematic_viscosity_real,
                                      truth_buoy_pile_real, usher_sampling,
                                      num_buoys)

        # find reward
        sampling.prepare_neighbor_and_boundary(runner, solver)
        simulation_v_real = sampling.sample_velocity(runner, solver)
        simulation_v_real.scale(unit.to_real_velocity(1))

        ground_truth.read_file(f'{ground_truth_dir}/v-{frame_id+1}.alu')
        reconstruction_error = runner.calculate_mse(simulation_v_real,
                                                    ground_truth,
                                                    n=sampling.num_samples)
        base = runner.calculate_mean_squared(ground_truth,
                                             sampling.num_samples)
        reward = -reconstruction_error / (base + 0.001)
        early_termination = False
        if reward < -5:
            print(f'early termination {reward} {do_nothing_reward}')
            early_termination = True

        for buoy_id in range(num_buoys):
            agent.remember(
                obs_aggregated[buoy_id], act_aggregated[buoy_id], reward,
                new_obs_aggregated[buoy_id],
                int(early_termination or frame_id == (num_frames - 2)))
        if sample_step >= agent.learn_after:  # as memory size is num_buoys * sample_step
            agent.learn()
        sample_step += 1
        score += reward
        obs_aggregated = new_obs_aggregated

        if dp.has_display():
            solver.normalize(solver.particle_v, particle_normalized_attr, 0,
                             unit.from_real_velocity(0.02))
        dp.unmap_graphical_pointers()
        if dp.has_display():
            display_proxy.draw()
        if early_termination:
            break
    score /= num_frames_in_scenario
    score_history.append(score)

    if episode_id % 50 == 0:
        agent.save_models(wandb.run.dir)
    wandb.log({'score': score, 'score100': np.mean(list(score_history))})
    dp.remove(ground_truth)
    sampling.destroy_variables()
    episode_id += 1
    if args.block_scan:
        break
