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
from util import Unit, FluidSample, parameterize_kinematic_viscosity, get_obs_dim, get_act_dim, make_obs, set_usher_param, get_coil_x_from_com, BuoySpec

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

# ground_truth_dir_list = [
#     f.path for f in os.scandir(args.truth_dir) if f.is_dir()
# ]

ground_truth_dir_list = [
    f"{args.truth_dir}/rltruth-0069a243-1224.19.26.30",
    f"{args.truth_dir}/rltruth-01846997-1226.23.14.08",
    f"{args.truth_dir}/rltruth-0344f9b6-1229.22.51.30",
    f"{args.truth_dir}/rltruth-066eb4c1-1225.22.19.14",
    f"{args.truth_dir}/rltruth-0dcf26ba-1227.23.39.36",
    f"{args.truth_dir}/rltruth-0edb3f6a-1227.13.26.06",
    f"{args.truth_dir}/rltruth-0f218c18-1226.04.02.53",
    f"{args.truth_dir}/rltruth-11c90dfd-1228.08.05.00",
    f"{args.truth_dir}/rltruth-13d6d1ae-1226.13.43.25",
    f"{args.truth_dir}/rltruth-16ebd14e-1228.05.50.28",
    f"{args.truth_dir}/rltruth-196e377b-1229.02.34.51",
    f"{args.truth_dir}/rltruth-1f86f7d5-1225.15.06.47",
    f"{args.truth_dir}/rltruth-1fbb1b4e-1227.19.01.28",
    f"{args.truth_dir}/rltruth-20698a89-1228.22.25.16",
    f"{args.truth_dir}/rltruth-20d1d20e-1224.23.04.45",
    f"{args.truth_dir}/rltruth-25b2076c-1229.22.27.51",
    f"{args.truth_dir}/rltruth-25ec96e0-1230.18.36.48",
    f"{args.truth_dir}/rltruth-28d07fd8-1228.16.16.04",
    f"{args.truth_dir}/rltruth-292456ca-1227.12.18.53",
    f"{args.truth_dir}/rltruth-29a8d5d5-1226.17.23.24",
    f"{args.truth_dir}/rltruth-2c467413-1224.21.08.50",
    f"{args.truth_dir}/rltruth-2dfae60a-1229.16.46.00",
    f"{args.truth_dir}/rltruth-2e952ebd-1229.15.06.08",
    f"{args.truth_dir}/rltruth-2ecd9057-1230.15.11.08",
    f"{args.truth_dir}/rltruth-3010c96c-1226.15.10.37",
    f"{args.truth_dir}/rltruth-3246bdca-1228.06.25.02",
    f"{args.truth_dir}/rltruth-33bb88b8-1225.20.22.34",
    f"{args.truth_dir}/rltruth-34e75942-1229.15.30.59",
    f"{args.truth_dir}/rltruth-36e4e29a-1226.22.49.23",
    f"{args.truth_dir}/rltruth-39a3a543-1228.18.27.34",
    f"{args.truth_dir}/rltruth-3a52eae0-1226.01.26.09",
    f"{args.truth_dir}/rltruth-3a8f7e8a-1225.21.42.56",
    f"{args.truth_dir}/rltruth-3c25c9ac-1228.04.52.28",
    f"{args.truth_dir}/rltruth-3c3255cd-1226.10.59.38",
    f"{args.truth_dir}/rltruth-3e6dcc78-1228.15.46.34",
    f"{args.truth_dir}/rltruth-3f019d49-1230.23.41.44",
    f"{args.truth_dir}/rltruth-3fbd49d6-1227.17.23.14",
    f"{args.truth_dir}/rltruth-4128e1b1-1230.02.23.03",
    f"{args.truth_dir}/rltruth-41f99e66-1224.17.47.20",
    f"{args.truth_dir}/rltruth-43c85dea-1225.18.35.20",
    f"{args.truth_dir}/rltruth-4853f4a5-1227.16.58.10",
    f"{args.truth_dir}/rltruth-49a206fd-1229.22.03.28",
    f"{args.truth_dir}/rltruth-4a8c8e29-1229.07.25.33",
    f"{args.truth_dir}/rltruth-4df5eb9c-1225.17.07.27",
    f"{args.truth_dir}/rltruth-515b7ca1-1227.07.56.06",
    f"{args.truth_dir}/rltruth-538e951c-1226.21.54.03",
    f"{args.truth_dir}/rltruth-58db752a-1230.14.40.17",
    f"{args.truth_dir}/rltruth-591db859-1231.00.15.21",
    f"{args.truth_dir}/rltruth-5b0e81fc-1225.09.58.11",
    f"{args.truth_dir}/rltruth-5bfe733a-1229.04.51.33",
    f"{args.truth_dir}/rltruth-5ef8805a-1226.19.37.35",
    f"{args.truth_dir}/rltruth-600aa9a7-1225.19.25.37",
    f"{args.truth_dir}/rltruth-61e76733-1226.17.53.05",
    f"{args.truth_dir}/rltruth-6369e230-1224.13.03.22",
    f"{args.truth_dir}/rltruth-68309e67-1229.11.44.00",
    f"{args.truth_dir}/rltruth-68d50727-1226.09.52.53",
    f"{args.truth_dir}/rltruth-6b66d754-1227.03.14.59",
    f"{args.truth_dir}/rltruth-6c81edf9-1227.05.35.41",
    f"{args.truth_dir}/rltruth-6d8ab276-1230.17.33.22",
    f"{args.truth_dir}/rltruth-6e0a4209-1228.01.12.39",
    f"{args.truth_dir}/rltruth-6e78494f-1225.16.00.06",
    f"{args.truth_dir}/rltruth-6f8866fe-1226.19.08.05",
    f"{args.truth_dir}/rltruth-6f8d0f73-1225.10.22.44",
    f"{args.truth_dir}/rltruth-7117c5bb-1228.09.06.25",
    f"{args.truth_dir}/rltruth-71280de7-1229.06.10.30",
    f"{args.truth_dir}/rltruth-72312cee-1226.14.43.52",
    f"{args.truth_dir}/rltruth-7329122a-1230.01.08.02",
    f"{args.truth_dir}/rltruth-7336fe3c-1225.10.48.57",
    f"{args.truth_dir}/rltruth-7373857b-1226.22.18.41",
    f"{args.truth_dir}/rltruth-7591fb0e-1230.20.22.56",
    f"{args.truth_dir}/rltruth-76a756ff-1228.02.38.52",
    f"{args.truth_dir}/rltruth-76d8b16c-1225.18.09.43",
    # f"{args.truth_dir}/rltruth-779d6e16-1230.09.28.15",
    # f"{args.truth_dir}/rltruth-779f3bd4-1224.17.18.53",
    # f"{args.truth_dir}/rltruth-7d95a88d-1225.03.35.14",
    # f"{args.truth_dir}/rltruth-7f803384-1227.07.26.03",
    # f"{args.truth_dir}/rltruth-805b6901-1229.08.01.01",
    # f"{args.truth_dir}/rltruth-81a71a1d-1226.06.58.14",
    # f"{args.truth_dir}/rltruth-825c4a8a-1226.13.15.09",
    # f"{args.truth_dir}/rltruth-84bd8e6c-1230.22.02.43",
    # f"{args.truth_dir}/rltruth-8607852d-1228.17.56.38",
    # f"{args.truth_dir}/rltruth-88da8322-1227.02.08.53",
    # f"{args.truth_dir}/rltruth-900ba381-1225.18.59.56",
    # f"{args.truth_dir}/rltruth-90ad4fc5-1229.05.16.43",
    # f"{args.truth_dir}/rltruth-91cbbb0b-1230.10.33.44",
    # f"{args.truth_dir}/rltruth-9281ad30-1227.20.16.27",
    # f"{args.truth_dir}/rltruth-950282ba-1230.05.28.31",
    # f"{args.truth_dir}/rltruth-95ee79e3-1230.12.15.58",
    # f"{args.truth_dir}/rltruth-a555b691-1230.12.44.57",
    # f"{args.truth_dir}/rltruth-a89ac95f-1229.14.30.57",
    # f"{args.truth_dir}/rltruth-a8f210d4-1228.21.03.12",
    # f"{args.truth_dir}/rltruth-a9ced4cc-1230.07.13.36",
    # f"{args.truth_dir}/rltruth-aab55f53-1227.09.45.53",
    # f"{args.truth_dir}/rltruth-ab088960-1224.18.11.34",
    # f"{args.truth_dir}/rltruth-adaf760f-1224.20.43.48",
    # f"{args.truth_dir}/rltruth-ae4de66f-1226.14.14.25",
    # f"{args.truth_dir}/rltruth-aeeaf562-1225.08.17.46",
    # f"{args.truth_dir}/rltruth-af146372-1227.00.53.56",
    # f"{args.truth_dir}/rltruth-afc9f413-1228.00.05.32",
    # f"{args.truth_dir}/rltruth-b0497bbc-1225.00.50.06",
    # f"{args.truth_dir}/rltruth-b07733a7-1230.22.26.43",
    # f"{args.truth_dir}/rltruth-b0b2e025-1228.08.30.53",
    # f"{args.truth_dir}/rltruth-b25413c5-1230.14.09.05",
    # f"{args.truth_dir}/rltruth-b44aad89-1227.08.30.54",
    # f"{args.truth_dir}/rltruth-b7bc9700-1226.11.25.02",
    # f"{args.truth_dir}/rltruth-b87dfc26-1228.04.22.57",
    # f"{args.truth_dir}/rltruth-b9bf8fd9-1229.11.03.27",
    # f"{args.truth_dir}/rltruth-ba850347-1230.21.38.00",
    # f"{args.truth_dir}/rltruth-bb4a59e1-1229.12.34.57",
    # f"{args.truth_dir}/rltruth-bc27ea2e-1228.18.52.17",
    # f"{args.truth_dir}/rltruth-be2f7051-1228.05.26.06",
    # f"{args.truth_dir}/rltruth-c01c83bd-1225.19.57.22",
    # f"{args.truth_dir}/rltruth-c02fca01-1228.21.34.27",
    # f"{args.truth_dir}/rltruth-c1014977-1224.19.53.55",
    # f"{args.truth_dir}/rltruth-c140c00d-1226.02.58.16",
    # f"{args.truth_dir}/rltruth-c89f2c3a-1225.13.09.58",
    # f"{args.truth_dir}/rltruth-c8e63dde-1229.14.06.23",
    # f"{args.truth_dir}/rltruth-c912da72-1227.21.54.29",
    # f"{args.truth_dir}/rltruth-c9ca9973-1228.01.46.37",
    # f"{args.truth_dir}/rltruth-ca9bdc80-1224.01.57.13",
    # f"{args.truth_dir}/rltruth-cc0957b9-1225.21.12.42",
    # f"{args.truth_dir}/rltruth-ccf24b01-1227.10.10.16",
    # f"{args.truth_dir}/rltruth-ce61f676-1224.16.03.47",
    # f"{args.truth_dir}/rltruth-d40ee8cc-1227.18.24.35",
    # f"{args.truth_dir}/rltruth-d860135c-1229.05.45.55",
    # f"{args.truth_dir}/rltruth-db355a14-1229.20.23.12",
    # f"{args.truth_dir}/rltruth-dbe8481f-1226.21.24.14",
    # f"{args.truth_dir}/rltruth-dc5a8311-1227.11.49.26",
    # f"{args.truth_dir}/rltruth-dd5eadbd-1230.13.37.24",
    # f"{args.truth_dir}/rltruth-e3326066-1230.05.58.26",
    # f"{args.truth_dir}/rltruth-e707c3ea-1226.03.26.58",
    # f"{args.truth_dir}/rltruth-e749248b-1228.19.16.55",
    # f"{args.truth_dir}/rltruth-e9887ee2-1225.16.37.09",
    # f"{args.truth_dir}/rltruth-e9e3f0bb-1230.16.57.40",
    # f"{args.truth_dir}/rltruth-ed209df3-1224.13.33.48",
    # f"{args.truth_dir}/rltruth-efcd9932-1225.06.28.28",
    # f"{args.truth_dir}/rltruth-f0dea7e2-1230.08.54.01",
    # f"{args.truth_dir}/rltruth-f12f02c8-1230.06.33.32",
    # f"{args.truth_dir}/rltruth-f28ef543-1225.14.40.37",
    # f"{args.truth_dir}/rltruth-f39032cb-1225.15.33.23",
    # f"{args.truth_dir}/rltruth-f3e7edad-1227.17.47.55",
    # f"{args.truth_dir}/rltruth-f718d97e-1225.13.42.32",
    # f"{args.truth_dir}/rltruth-f8052bd0-1226.16.08.24",
    # f"{args.truth_dir}/rltruth-f9efd691-1225.17.37.56",
    # f"{args.truth_dir}/rltruth-fb640cf5-1229.10.31.00",
    # f"{args.truth_dir}/rltruth-fd77be22-1227.03.50.37",
    # f"{args.truth_dir}/rltruth-fdccb666-1226.15.38.36",
    # f"{args.truth_dir}/rltruth-fe5f814e-1225.12.36.09",
]

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
            learn_after=10000,
            replay_size=36000000,
            hidden_sizes=[2048, 2048, 1024],
            actor_final_scale=1,
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
    mask = dp.create_coated((sampling.num_samples), 1, np.uint32)
    sampling.sample_x.scale(unit.from_real_length(1))
    ground_truth = dp.create_coated_like(sampling.sample_data3)
    sum_v2 = unit.from_real_velocity_mse(
        np.load(f'{ground_truth_dir}/sum_v2.npy').item())
    max_v2 = unit.from_real_velocity_mse(
        np.load(f'{ground_truth_dir}/max_v2.npy').item())
    dp.map_graphical_pointers()
    solver.update_particle_neighbors()
    num_buoys = dp.Pile.get_size_from_file(f'{ground_truth_dir}/0.pile') - 2
    truth_buoy_pile_real.read_file(f'{ground_truth_dir}/0.pile', num_buoys, 0,
                                   1)
    coil_x_real = get_coil_x_from_com(dp, unit, buoy_spec,
                                      truth_buoy_pile_real, num_buoys)
    # set positions for sampling around buoys in simulation
    coil_x_np = unit.from_real_length(coil_x_real)
    usher_sampling.sample_x.set(coil_x_np)
    usher_sampling.prepare_neighbor_and_boundary(runner, solver)
    usher_sampling.sample_density(runner)
    usher_sampling.sample_velocity(runner, solver)
    usher_sampling.sample_vorticity(runner, solver)

    obs_aggregated = make_obs(dp, unit, kinematic_viscosity_real,
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
                                       num_buoys, 0, 1)
        coil_x_real = get_coil_x_from_com(dp, unit, buoy_spec,
                                          truth_buoy_pile_real, num_buoys)
        coil_x_np = unit.from_real_length(coil_x_real)
        usher_sampling.sample_x.set(coil_x_np)

        if sample_step < agent.learn_after:
            act_aggregated = np.zeros((num_buoys, agent.act_dim))
            for buoy_id in range(num_buoys):
                act_aggregated[buoy_id] = agent.uniform_random_action()
        else:
            act_aggregated = agent.get_action(obs_aggregated,
                                              enable_noise=not validation_mode)
        if np.sum(np.isnan(act_aggregated)) > 0:
            print(obs_aggregated, act_aggregated)
            sys.exit(0)
        act_aggregated_converted = agent.actor.from_normalized_action(
            act_aggregated)
        set_usher_param(solver.usher, dp, unit, truth_buoy_pile_real,
                        coil_x_real, act_aggregated_converted, num_buoys)
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
                                       num_buoys, 0, 1)
        coil_x_real = get_coil_x_from_com(dp, unit, buoy_spec,
                                          truth_buoy_pile_real, num_buoys)
        coil_x_np = unit.from_real_length(coil_x_real)
        usher_sampling.sample_x.set(
            coil_x_np)  # NOTE: should set to new sampling points?
        usher_sampling.prepare_neighbor_and_boundary(runner, solver)
        usher_sampling.sample_density(runner)
        usher_sampling.sample_velocity(runner, solver)
        usher_sampling.sample_vorticity(runner, solver)
        new_obs_aggregated = make_obs(dp, unit, kinematic_viscosity_real,
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
                    obs_aggregated[buoy_id], act_aggregated[buoy_id], reward,
                    new_obs_aggregated[buoy_id],
                    int(early_termination or frame_id == (num_frames - 2)))
            if sample_step >= agent.learn_after:  # as memory size is num_buoys * sample_step
                if not dumped_buffer:
                    agent.memory.save('buffer-dump')
                    dumped_buffer = True
                agent.learn()
        else:
            # NOTE: Saving all required data for animation. Only for validating a single scenario.
            np.save(f'val/act-{frame_id}.npy', act_aggregated_converted)
            np.save(f'val/obs-{frame_id}.npy', obs_aggregated)
            buoys_q0, buoys_q1 = agent.get_value(obs_aggregated,
                                                 act_aggregated)
            np.save(f'val/value0-{frame_id}.npy',
                    buoys_q0.cpu().detach().numpy())
            np.save(f'val/value1-{frame_id}.npy',
                    buoys_q1.cpu().detach().numpy())
            simulation_v_real.write_file(f'val/v-{frame_id}.alu',
                                         sampling.num_samples)
            np.save(f'val/reward-{frame_id}.npy', reward)

        sample_step += 1
        obs_aggregated = new_obs_aggregated

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
    dp.remove(mask)
    dp.remove(ground_truth)
    dp.remove(visual_x_scaled)
    dp.remove(ground_truth)
    sampling.destroy_variables()
    episode_id += 1
    if args.block_scan:
        break
    if validation_mode:
        break
