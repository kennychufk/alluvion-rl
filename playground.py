import argparse
import random
from collections import deque

import alluvion as al
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error
import wandb
import torch

from ddpg_torch import DDPGAgent

parser = argparse.ArgumentParser(description='RL playground')
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--input', type=str, default='')
args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

class FluidField:
    def __init__(self, dp, field_box_min, field_box_max):
        self.dp = dp
        self.num_samples_per_dim = 24
        self.num_samples = self.num_samples_per_dim * self.num_samples_per_dim * self.num_samples_per_dim
        self.sample_x = dp.create_coated((self.num_samples), 3)
        self.sample_data3 = dp.create_coated((self.num_samples), 3)
        self.sample_neighbors = dp.create_coated(
            (self.num_samples, dp.cni.max_num_neighbors_per_particle), 4)
        self.sample_num_neighbors = dp.create_coated((self.num_samples), 1,
                                                     np.uint32)

        self.sample_x_host = np.zeros((self.num_samples, 3), dp.default_dtype)

        field_box_size = dp.f3(field_box_max.x - field_box_min.x,
                               field_box_max.y - field_box_min.y,
                               field_box_max.z - field_box_min.z)
        for i in range(self.num_samples):
            z_id = i % self.num_samples_per_dim
            y_id = i % (self.num_samples_per_dim *
                        self.num_samples_per_dim) // self.num_samples_per_dim
            x_id = i // (self.num_samples_per_dim * self.num_samples_per_dim)
            self.sample_x_host[i] = np.array([
                field_box_min.x + field_box_size.x /
                (self.num_samples_per_dim - 1) * x_id, field_box_min.y +
                field_box_size.y / (self.num_samples_per_dim - 1) * y_id,
                field_box_min.z + field_box_size.z /
                (self.num_samples_per_dim - 1) * z_id
            ])
        self.sample_x.set(self.sample_x_host)


# marker v, marker q, marker omega, fluid v, fluid density, TODO: distance to boundary, normal to boundary
def make_obs(marker_v, marker_q, marker_omega, sample_v, sample_density):
    return np.concatenate(
        (marker_v.flatten(), marker_q.flatten(), marker_omega.flatten(),
         sample_v.flatten(), sample_density),
        axis=None,
        dtype=marker_v.dtype)


def make_usher_param(marker_x, marker_v, act_aggregated, kernel_radius):
    drive_x = marker_x  # TODO: can be different
    drive_v = marker_v  # TODO: can be different
    num_buoys = len(marker_x)
    drive_radius = np.ascontiguousarray(act_aggregated[:num_buoys] *
                                        kernel_radius)
    drive_strength = np.ascontiguousarray(act_aggregated[num_buoys:])
    return drive_x, drive_v, drive_radius, drive_strength


dp = al.Depot(np.float32)
cn = dp.cn
cni = dp.cni
dp.create_display(800, 600, "", False)
display_proxy = dp.get_display_proxy()
runner = dp.Runner()

particle_radius = 2**-9
kernel_radius = particle_radius * 4
density0 = 1000.0
cubical_particle_volume = 8 * particle_radius * particle_radius * particle_radius
volume_relative_to_cube = 0.8
particle_mass = cubical_particle_volume * volume_relative_to_cube * density0
gravity = dp.f3(0, -9.81, 0)

cn.set_cubic_discretization_constants()
cn.set_kernel_radius(kernel_radius)
cn.set_particle_attr(particle_radius, particle_mass, density0)
cn.boundary_epsilon = 1e-9
cn.gravity = gravity
cn.viscosity = 2.15217905e-05
cn.boundary_viscosity = 1.72357061e-05

# rigids
max_num_contacts = 512
pile = dp.Pile(dp, max_num_contacts)

container_width = 0.24
container_dim = dp.f3(container_width, container_width, container_width)
container_mesh = al.Mesh()
container_mesh.set_box(container_dim, 8)
container_distance = dp.BoxDistance(container_dim)
pile.add(container_distance,
         al.uint3(96, 96, 96),
         sign=-1,
         thickness=(2**-9),
         collision_mesh=container_mesh,
         mass=0,
         restitution=0.8,
         friction=0.2,
         inertia_tensor=dp.f3(1, 1, 1),
         x=dp.f3(0, container_width / 2, 0),
         q=dp.f4(0, 0, 0, 1),
         display_mesh=al.Mesh())

inset_factor = 1.71

pile.build_grids(4 * kernel_radius)
pile.reallocate_kinematics_on_device()
pile.set_gravity(gravity)
cn.contact_tolerance = particle_radius

block_mode = 0
edge_factor = 0.49
max_num_particles = int(4232896 / 64)
print('num_particles', max_num_particles)
grid_side = int(np.ceil((container_width + kernel_radius * 2) / kernel_radius))
grid_side += (grid_side % 2 == 1)
# grid_side = 64
grid_res = al.uint3(grid_side, grid_side, grid_side)
print('grid_res', grid_res)
grid_offset = al.int3(-grid_side // 2, -1, -grid_side // 2)

cni.grid_res = grid_res
cni.grid_offset = grid_offset
cni.max_num_particles_per_cell = 64
cni.max_num_neighbors_per_particle = 64

num_buoys = 7
solver = dp.SolverDf(runner,
                     pile,
                     dp,
                     max_num_particles,
                     grid_res,
                     num_ushers=num_buoys,
                     enable_surface_tension=False,
                     enable_vorticity=False,
                     graphical=True)
particle_normalized_attr = dp.create_graphical((max_num_particles), 1)

solver.dt = 1e-3
solver.max_dt = particle_radius * 0.2
solver.min_dt = 0.0
solver.cfl = 0.15
solver.particle_radius = particle_radius
solver.num_particles = max_num_particles

dp.copy_cn()

dp.map_graphical_pointers()
if len(args.input) == 0:
    # runner.launch_create_fluid_cylinder_sunflower(256, solver.particle_x, max_num_particles, radius=(container_width *0.5) - kernel_radius, num_particles_per_slice=400, slice_distance = particle_radius * 2, y_min=kernel_radius)
    runner.launch_create_fluid_cylinder_sunflower(
        256,
        solver.particle_x,
        max_num_particles,
        radius=(container_width * 0.5) - kernel_radius,
        num_particles_per_slice=1600,
        slice_distance=particle_radius * 2,
        y_min=kernel_radius)
else:
    solver.particle_x.read_file(f'{args.input}-x.alu')
    solver.particle_v.read_file(f'{args.input}-v.alu')
dp.unmap_graphical_pointers()
display_proxy.set_camera(al.float3(0, 0.06, 0.4), al.float3(0, 0.06, 0))
colormap_tex = display_proxy.create_colormap_viridis()

display_proxy.add_particle_shading_program(solver.particle_x,
                                           particle_normalized_attr,
                                           colormap_tex,
                                           solver.particle_radius, solver)
display_proxy.add_pile_shading_program(pile)

next_force_time = 0.0
remaining_force_time = 0.0

# frame_directory = 'rl-truth-6a31d4'
# Path(frame_directory).mkdir(parents=True, exist_ok=True)

agent = DDPGAgent(actor_lr=2e-5,
                  critic_lr=2e-4,
                  critic_weight_decay=1e-2,
                  obs_dim=14 * num_buoys,
                  act_dim=2 * num_buoys,
                  hidden_sizes=[2048, 1800],
                  soft_update_rate=0.001,
                  batch_size=64,
                  final_layer_magnitude=1e-4)
field_box_min = dp.f3(container_width * -0.5, 0, container_width * -0.5)
field_box_max = dp.f3(container_width * 0.5, container_width,
                      container_width * 0.5)
field = FluidField(dp, field_box_min, field_box_max)

fps = 60.0
frame_interval = 1.0 / fps
next_frame_id = 0
ground_truth_pile = dp.Pile(dp, 0)

wandb.init(project='alluvion-rl')
config = wandb.config
config.actor_lr = agent.actor_lr
config.critic_lr = agent.critic_lr
config.critic_weight_decay = agent.critic_weight_decay
config.obs_dim = agent.obs_dim
config.act_dim = agent.act_dim
config.hidden_sizes = agent.hidden_sizes
config.soft_update_rate = agent.soft_update_rate
config.gamma = agent.gamma
config.replay_size = agent.replay_size
config.final_layer_magnitude = agent.final_layer_magnitude
config.seed = args.seed

wandb.watch(agent.critic)

with open('switch', 'w') as f:
    f.write('1')

ground_truth_dir_list = ['./rl-truth-e6a7da','./rl-truth-6a31d4', './rl-truth-f2caa5', './rl-truth-ed39a2']
score_history = deque(maxlen=100)
episode_id = 0
while True:
    with open('switch', 'r') as f:
        if f.read(1) == '0':
            break
    dir_id = random.randrange(len(ground_truth_dir_list))
    ground_truth_dir = ground_truth_dir_list[dir_id]
    dp.map_graphical_pointers()
    solver.reset_solving_var()
    solver.t = 0
    solver.particle_x.read_file(f'{args.input}-x.alu')
    solver.particle_v.read_file(f'{args.input}-v.alu')
    solver.update_particle_neighbors()
    dp.unmap_graphical_pointers()
    gtx, gtv, gtq, gtomega = dp.read_pile(f'{ground_truth_dir}/0.pile',
                                          num_buoys + 2)
    solver.usher.set_sample_x(gtx[1:1 + num_buoys])
    solver.sample_usher()
    obs = make_obs(gtv[1:1 + num_buoys], gtq[1:1 + num_buoys],
                   gtomega[1:1 + num_buoys],
                   dp.coat(solver.usher.sample_v).get(),
                   dp.coat(solver.usher.sample_density).get())
    # clear usher
    drive_x = gtx[1:1 + num_buoys]
    drive_v = np.repeat(0, num_buoys * 3).astype(dp.default_dtype)
    drive_radius = np.repeat(kernel_radius, num_buoys).astype(dp.default_dtype)
    drive_strength = np.repeat(0, num_buoys).astype(dp.default_dtype)
    solver.usher.set(drive_x, drive_v, drive_radius, drive_strength)
    score = 0
    with open(f'{ground_truth_dir}/range.txt', 'r') as f:
        num_frames = int(f.readline())
        print('num_frames = ', num_frames)
    for frame_id in range(num_frames - 1):
        target_t = frame_id / fps
        gtx, gtv, gtq, gtomega = dp.read_pile(
            f'{ground_truth_dir}/{frame_id}.pile', num_buoys + 2)
        solver.usher.set_sample_x(gtx[1:1 + num_buoys])
        act_aggregated = agent.get_action(obs)
        solver.usher.set(*make_usher_param(gtx[1:1 + num_buoys], gtv[1:1 +
                                                                     num_buoys],
                                           act_aggregated, kernel_radius))
        dp.map_graphical_pointers()
        while (solver.t < target_t):
            solver.step()
            # TODO: do sub-frame choose_action
        print(frame_id)
        gtx, gtv, gtq, gtomega = dp.read_pile(
            f'{ground_truth_dir}/{frame_id+1}.pile', num_buoys + 2)
        new_obs = make_obs(gtv[1:1 + num_buoys], gtq[1:1 + num_buoys],
                           gtomega[1:1 + num_buoys],
                           dp.coat(solver.usher.sample_v).get(),
                           dp.coat(solver.usher.sample_density).get())

        # find reward
        solver.update_particle_neighbors()
        runner.launch_make_neighbor_list(field.sample_x, solver.pid,
                                         solver.pid_length, field.sample_neighbors,
                                         field.sample_num_neighbors,
                                         field.num_samples)
        runner.launch_sample_fluid(field.sample_x, solver.particle_x,
                                   solver.particle_density, solver.particle_v,
                                   field.sample_neighbors,
                                   field.sample_num_neighbors, field.sample_data3,
                                   field.num_samples)
        # TODO: accelerate using CUDA kernel
        simulated_v = field.sample_data3.get()
        field.sample_data3.read_file(f'{ground_truth_dir}/vfield-{frame_id+1}.alu')
        ground_truth_v = field.sample_data3.get()
        reward = -mean_squared_error(simulated_v, ground_truth_v)

        agent.remember(obs, act_aggregated, reward, new_obs,
                       int(frame_id == (num_frames - 2)))
        agent.learn()
        score += reward
        obs = new_obs

        solver.normalize(solver.particle_v, particle_normalized_attr, 0, 2)
        dp.unmap_graphical_pointers()
        display_proxy.draw()
    score_history.append(score)

    if episode_id % 50 == 0:
        agent.save_models(wandb.run.dir)
    wandb.log({'score': score, 'score100': np.mean(list(score_history))})
    episode_id += 1
