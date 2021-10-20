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
from util import Unit, FluidSample, get_timestamp_and_hash

parser = argparse.ArgumentParser(description='RL playground')
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--initial', type=str, default='')
parser.add_argument('--output-dir', type=str, default='.')
parser.add_argument('--truth-dir', type=str, default='.')
args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)


# using real unit except density: which is relative to density0
def make_obs(dp, unit, truth_buoy_pile_real, usher_sampling, num_buoys):
    obs_aggregated = np.zeros([num_buoys, agent.obs_dim], dp.default_dtype)
    buoy_x_real = dp.coat(truth_buoy_pile_real.x).get()
    buoy_v_real = dp.coat(truth_buoy_pile_real.v).get()
    buoy_q = dp.coat(truth_buoy_pile_real.q).get()

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
            (xi, buoy_v_real[buoy_id], buoy_q[buoy_id],
             xij[dist_sort_index[0]], vi - buoy_v_real[dist_sort_index[0]],
             xij[dist_sort_index[1]], vi - buoy_v_real[dist_sort_index[1]],
             sample_v_real[buoy_id].flatten(),
             sample_density_relative[buoy_id],
             sample_vort_real[buoy_id].flatten(),
             sample_container_kernel_vol_grad_real[buoy_id].flatten(),
             sample_container_kernel_vol[buoy_id].flatten()),
            axis=None)
    return obs_aggregated


def set_usher_param(usher, dp, unit, truth_buoy_pile_real, act_aggregated):
    # [0:3] [3:6] [6:9] displacement from buoy x
    # [9:12] [12:15] [15:18] velocity offset from buoy v
    # [18] focal dist
    # [19] usher kernel radius
    # [20] strength
    xoffset_real = np.zeros((3, num_buoys, 3), dp.default_dtype)
    voffset_real = np.zeros((3, num_buoys, 3), dp.default_dtype)
    # xoffset_real = act_aggregated[:, 0:9].reshape(3, num_buoys,  3);
    # voffset_real = act_aggregated[:, 9:18].reshape(3, num_buoys,  3);
    # print(xoffset_real, voffset_real)
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

    dp.coat(usher.focal_dist).set(np.ones(num_buoys, dp.default_dtype) * 6)
    # dp.coat(usher.focal_dist).set(unit.from_real_length(act_aggregated[:, 0]))

    dp.coat(usher.usher_kernel_radius).set(
        np.ones(num_buoys, dp.default_dtype) * 2)
    # dp.coat(usher.usher_kernel_radius).set(unit.from_real_length(np.ones(num_buoys, dp.default_dtype) * 0.04))
    # dp.coat(usher.usher_kernel_radius).set(
    #     unit.from_real_length(act_aggregated[:, 0]))

    # dp.coat(usher.drive_strength).set(np.ones(num_buoys, dp.default_dtype) * 2)
    # DEBUG
    dp.coat(usher.drive_strength).set(
        unit.from_real_angular_velocity(act_aggregated[:, 0]))
    # dp.coat(usher.drive_strength).set(
    #     unit.from_real_angular_velocity(np.ones_like(act_aggregated[:, 0]) * 100))

    # print('buoy_x_real', buoy_x_real)
    # print('xoffset_real', xoffset_real)
    # print('focal_x', dp.coat(usher.focal_x).get())

    # print('buoy_v_real', buoy_v_real)
    # print('voffset_real', voffset_real)
    # print('focal_v', dp.coat(usher.focal_v).get())

    # print('focal_dist', dp.coat(usher.focal_dist).get())
    # print('usher_kernel_radius', dp.coat(usher.usher_kernel_radius).get())
    # print('drive_strength', dp.coat(usher.drive_strength).get())


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

unit = Unit(real_kernel_radius=0.020,
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
         thickness=unit.from_real_length(0.005),
         collision_mesh=container_mesh,
         mass=0,
         restitution=0.8,
         friction=0.2,
         inertia_tensor=dp.f3(1, 1, 1),
         x=dp.f3(0, container_width / 2, 0),
         q=dp.f4(0, 0, 0, 1),
         display_mesh=al.Mesh())

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
                (container_width * 2 - kernel_radius * 4) * edge_factor,
                (container_width - kernel_radius * 2) * edge_factor)
fluid_mass = unit.from_real_mass(2.77235)
num_particles = int(fluid_mass / cn.particle_mass)
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

num_buoys = 8
generating_initial = (len(args.initial) == 0)
solver = dp.SolverI(runner,
                    pile,
                    dp,
                    num_particles,
                    cni.grid_res,
                    num_ushers=0 if generating_initial else num_buoys,
                    enable_surface_tension=False,
                    enable_vorticity=False,
                    graphical=True)
particle_normalized_attr = dp.create_graphical((num_particles), 1)

usher_sampling = FluidSample(dp, np.zeros((num_buoys, 3), dp.default_dtype))

solver.num_particles = num_particles
solver.max_dt = unit.from_real_time(0.05 * unit.rl)
solver.initial_dt = solver.max_dt
solver.min_dt = 0
solver.cfl = 0.4

dp.copy_cn()

dp.map_graphical_pointers()
if generating_initial:
    runner.launch_create_fluid_block(solver.particle_x,
                                     solver.num_particles,
                                     offset=0,
                                     particle_radius=particle_radius,
                                     mode=block_mode,
                                     box_min=box_min,
                                     box_max=box_max)
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

score_history = deque(maxlen=100)
episode_id = 0

timestamp_str, timestamp_hash = get_timestamp_and_hash()
if generating_initial:
    initial_directory = f'{args.output_dir}/reconinit-{timestamp_hash}-{timestamp_str}'
    Path(initial_directory).mkdir(parents=True, exist_ok=True)

    with open('switch', 'w') as f:
        f.write('1')
    while True:
        dp.map_graphical_pointers()
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
            solver.step()
        dp.unmap_graphical_pointers()
        display_proxy.draw()
    sys.exit(0)

ground_truth_dir_list = [
    f.path for f in os.scandir(args.truth_dir) if f.is_dir()
]

truth_real_freq = 100.0
truth_real_interval = 1.0 / truth_real_freq
next_truth_frame_id = 0
truth_buoy_pile_real = dp.Pile(dp, runner, 0)
for i in range(num_buoys):
    truth_buoy_pile_real.add(dp.SphereDistance.create(0), al.uint3(64, 64, 64))

agent = TD3(
    actor_lr=3e-4,
    critic_lr=3e-4,
    critic_weight_decay=0,
    obs_dim=33,
    # act_dim=21,
    act_dim=1,
    expl_noise_func=GaussianNoise(std_dev=0.1),
    gamma=0.95,
    # hidden_sizes=[2048, 1800],
    min_action=np.array([0]),
    max_action=np.array([1000]),
    learn_after=100000,
    hidden_sizes=[600, 400],
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
    # dir_id = random.randrange(len(ground_truth_dir_list))
    dir_id = 0
    ground_truth_dir = ground_truth_dir_list[dir_id]
    print(dir_id, ground_truth_dir)
    sampling = FluidSample(dp, f'{ground_truth_dir}/sample-x.alu')
    sampling.sample_x.scale(unit.from_real_length(1))
    dp.map_graphical_pointers()
    solver.reset_solving_var()
    solver.t = 0
    solver.particle_x.read_file(f'{args.initial}/x.alu')
    solver.particle_v.read_file(f'{args.initial}/v.alu')
    solver.particle_pressure.read_file(f'{args.initial}/pressure.alu')
    solver.update_particle_neighbors()
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

    obs_aggregated = make_obs(dp, unit, truth_buoy_pile_real, usher_sampling,
                              num_buoys)

    dp.unmap_graphical_pointers()
    # # clear usher
    # initial_strength = np.zeros(num_buoys, dp.default_dtype)
    # set_usher_param(solver.usher, dp, unit, truth_buoy_pile_real,
    #                 initial_strength)

    score = 0
    do_nothing_score = 0

    num_frames = 1000
    truth_real_freq = 100.0
    truth_real_interval = 1.0 / truth_real_freq
    for frame_id in range(num_frames - 1):
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
        if frame_id % 100 == 0:
            print(frame_id, act_aggregated)
        if np.sum(np.isnan(act_aggregated)) > 0:
            print(obs_aggregated, act_aggregated)
            sys.exit(0)
        set_usher_param(solver.usher, dp, unit, truth_buoy_pile_real,
                        agent.actor.from_normalized_action(act_aggregated))
        dp.map_graphical_pointers()
        while (solver.t < target_t):
            solver.step()
        # print(frame_id)
        truth_buoy_pile_real.read_file(f'{ground_truth_dir}/{frame_id+1}.pile',
                                       num_buoys, 0, 1)
        usher_sampling.prepare_neighbor_and_boundary(runner, solver)
        usher_sampling.sample_density(runner)
        usher_sampling.sample_velocity(runner, solver)
        usher_sampling.sample_vorticity(runner, solver)
        new_obs_aggregated = make_obs(dp, unit, truth_buoy_pile_real,
                                      usher_sampling, num_buoys)

        # find reward
        sampling.prepare_neighbor_and_boundary(runner, solver)
        simulation_v_real = sampling.sample_velocity(runner, solver)
        simulation_v_real.scale(unit.to_real_velocity(1))

        # TODO: accelerate using CUDA kernel
        simulation_v_real_np = simulation_v_real.get()
        sampling.sample_data3.read_file(
            f'{ground_truth_dir}/v-{frame_id+1}.alu')
        truth_v_real_np = sampling.sample_data3.get()
        reward = -mean_squared_error(simulation_v_real_np, truth_v_real_np)
        do_nothing_reward = -mean_squared_error(np.zeros_like(truth_v_real_np),
                                                truth_v_real_np)
        early_termination = False
        if reward < do_nothing_reward * 10:
            print(f'early termination {reward} {do_nothing_reward}')
            early_termination = True

        for buoy_id in range(num_buoys):
            agent.remember(
                obs_aggregated[buoy_id], act_aggregated[buoy_id], reward * 10,
                new_obs_aggregated[buoy_id],
                int(early_termination or frame_id == (num_frames - 2)))
        if sample_step >= agent.learn_after:  # as memory size is num_buoys * sample_step
            agent.learn()
        sample_step += 1
        score += reward
        do_nothing_score += do_nothing_reward
        obs_aggregated = new_obs_aggregated

        solver.normalize(solver.particle_v, particle_normalized_attr, 0,
                         unit.from_real_velocity(0.02))
        dp.unmap_graphical_pointers()
        display_proxy.draw()
        if early_termination:
            break
    score_history.append(score)

    if episode_id % 50 == 0:
        agent.save_models(wandb.run.dir)
    wandb.log({
        'score': score,
        'offset_score': (score - do_nothing_score),
        'score100': np.mean(list(score_history))
    })
    sampling.destroy_variables()
    episode_id += 1
