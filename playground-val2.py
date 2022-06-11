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
import time

from rl import TD3, OrnsteinUhlenbeckProcess, GaussianNoise
from util import Unit, FluidSamplePellets, parameterize_kinematic_viscosity_with_pellets, get_state_dim, get_action_dim, make_state, set_usher_param, get_coil_x_from_com, BuoySpec, read_pile, get_timestamp_and_hash

parser = argparse.ArgumentParser(description='RL playground')
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--cache-dir', type=str, default='.')
parser.add_argument('--truth-dir', type=str, required=True)
parser.add_argument('--display', metavar='d', type=bool, default=False)
parser.add_argument('--model-dir', type=str, required=True)
args = parser.parse_args()


class Environment:

    def __init__(self, dp, truth_dirs, display):
        self.dp = dp
        self.cn = self.dp.cn
        cni = self.dp.cni
        self.display = display
        if display:
            self.dp.create_display(800, 600, "", False)
        self.display_proxy = self.dp.get_display_proxy() if display else None
        self.runner = self.dp.Runner()

        # === constants
        particle_radius = 0.25
        kernel_radius = 1.0
        density0 = 1.0
        cubical_particle_volume = 8 * particle_radius * particle_radius * particle_radius
        volume_relative_to_cube = 0.8
        particle_mass = cubical_particle_volume * volume_relative_to_cube * density0

        gravity = self.dp.f3(0, -1, 0)

        real_kernel_radius = 2**-6
        self.unit = Unit(
            real_kernel_radius=real_kernel_radius,
            real_density0=1,  # dummy
            real_gravity=-9.80665)

        self.cn.set_kernel_radius(kernel_radius)
        self.cn.set_particle_attr(particle_radius, particle_mass, density0)
        self.cn.boundary_epsilon = 1e-9
        self.cn.gravity = gravity
        self.truth_dirs = truth_dirs
        # === constants

        max_num_buoys = 0
        max_num_beads = 0
        for truth_dir in truth_dirs:
            unit = Unit(
                real_kernel_radius=self.unit.rl,
                real_density0=np.load(f'{truth_dir}/density0_real.npy').item(),
                real_gravity=self.unit.rg)
            fluid_mass = unit.from_real_mass(
                np.load(f'{truth_dir}/fluid_mass.npy').item())
            num_beads = int(fluid_mass / self.cn.particle_mass)
            if num_beads > max_num_beads:
                max_num_beads = num_beads
            num_buoys = self.dp.Pile.get_size_from_file(
                f'{truth_dir}/0.pile') - 2
            if num_buoys > max_num_buoys:
                max_num_buoys = num_buoys
        print('max_num_buoys', max_num_buoys)
        print('max_num_beads', max_num_beads)

        container_pellet_filename = '/home/kennychufk/workspace/pythonWs/alluvion-optim/cube24-2to-6.alu'
        container_num_pellets = self.dp.get_alu_info(
            container_pellet_filename)[0][0]

        # rigids
        self.pile = self.dp.Pile(self.dp,
                                 self.runner,
                                 max_num_contacts=0,
                                 volume_method=al.VolumeMethod.pellets,
                                 max_num_pellets=container_num_pellets)

        ## ================== container
        container_width = self.unit.from_real_length(0.24)
        container_dim = self.dp.f3(container_width, container_width,
                                   container_width)
        container_distance = self.dp.BoxDistance.create(container_dim,
                                                        outset=0)
        container_extent = container_distance.aabb_max - container_distance.aabb_min
        container_res_float = container_extent / particle_radius
        container_res = al.uint3(int(container_res_float.x),
                                 int(container_res_float.y),
                                 int(container_res_float.z))
        print('container_res', container_res)
        container_pellet_x = self.dp.create((container_num_pellets), 3)
        container_pellet_x.read_file(container_pellet_filename)
        self.pile.add_pellets(container_distance,
                              container_res,
                              pellets=container_pellet_x,
                              sign=-1,
                              mass=0,
                              restitution=0.8,
                              friction=0.3)
        self.dp.remove(container_pellet_x)
        ## ================== container
        self.buoy_spec = BuoySpec(self.dp, self.unit)

        self.pile.reallocate_kinematics_on_device()
        self.pile.set_gravity(gravity)
        self.cn.contact_tolerance = particle_radius

        container_aabb_range_per_h = container_extent / kernel_radius
        cni.grid_res = al.uint3(int(math.ceil(container_aabb_range_per_h.x)),
                                int(math.ceil(container_aabb_range_per_h.y)),
                                int(math.ceil(
                                    container_aabb_range_per_h.z))) + 4
        cni.grid_offset = al.int3(
            int(container_distance.aabb_min.x) - 2,
            int(container_distance.aabb_min.y) - 2,
            int(container_distance.aabb_min.z) - 2)
        cni.max_num_particles_per_cell = 64
        cni.max_num_neighbors_per_particle = 64

        self._max_episode_steps = 1000
        self.solver = self.dp.SolverI(self.runner,
                                      self.pile,
                                      self.dp,
                                      max_num_particles=max_num_beads,
                                      num_ushers=max_num_buoys,
                                      enable_surface_tension=False,
                                      enable_vorticity=False,
                                      graphical=display)
        self.solver.max_dt = self.unit.from_real_time(0.0005)
        self.solver.initial_dt = self.solver.max_dt
        self.solver.min_dt = 0
        self.solver.cfl = 0.2
        self.usher_sampling = FluidSamplePellets(
            self.dp, np.zeros((max_num_buoys, 3), self.dp.default_dtype))
        self.dir_id = 0
        self.simulation_sampling = None
        self.ground_truth_v = None
        self.mask = None

        if display:
            self.display_proxy.set_camera(
                self.unit.from_real_length(al.float3(0, 0.06, 0.4)),
                self.unit.from_real_length(al.float3(0, 0.0, 0)))
            self.display_proxy.set_clip_planes(
                self.unit.to_real_length(1),
                container_distance.aabb_max.z * 20)
            colormap_tex = self.display_proxy.create_colormap_viridis()
            self.particle_normalized_attr = self.dp.create_graphical(
                (max_num_beads), 1)

            self.display_proxy.add_particle_shading_program(
                self.solver.particle_x, self.particle_normalized_attr,
                colormap_tex, self.solver.particle_radius, self.solver)
            self.display_proxy.add_pile_shading_program(self.pile)

    def seed(self, seed_num):
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    def reset(self):
        if self.simulation_sampling is not None:
            self.simulation_sampling.destroy_variables()
        if self.ground_truth_v is not None:
            self.dp.remove(self.ground_truth_v)
        if self.mask is not None:
            self.dp.remove(self.mask)
        self.truth_dir = self.truth_dirs[self.dir_id]
        self.dir_id = (self.dir_id + 1) % len(self.truth_dirs)
        print(self.dir_id, self.truth_dir)

        self.unit = Unit(real_kernel_radius=self.unit.rl,
                         real_density0=np.load(
                             f'{self.truth_dir}/density0_real.npy').item(),
                         real_gravity=self.unit.rg)
        self.kinematic_viscosity_real = np.load(
            f'{self.truth_dir}/kinematic_viscosity_real.npy').item()

        self.cn.set_particle_attr(self.cn.particle_radius,
                                  self.cn.particle_mass, self.cn.density0)
        self.cn.viscosity, self.cn.boundary_viscosity = self.unit.from_real_kinematic_viscosity(
            parameterize_kinematic_viscosity_with_pellets(
                self.kinematic_viscosity_real))

        fluid_mass = self.unit.from_real_mass(
            np.load(f'{self.truth_dir}/fluid_mass.npy').item())
        num_particles = int(fluid_mass / self.cn.particle_mass)
        self.solver.num_particles = num_particles

        self.num_buoys = self.dp.Pile.get_size_from_file(
            f'{self.truth_dir}/0.pile') - 2
        self.solver.usher.reset()
        self.solver.usher.num_ushers = self.num_buoys
        print('num_particles', num_particles, 'self.num_buoys', self.num_buoys)

        initial_particle_x_filename = f'{args.cache_dir}/playground_bead_x-2to-6.alu'
        initial_particle_v_filename = f'{args.cache_dir}/playground_bead_v-2to-6.alu'
        initial_particle_pressure_filename = f'{args.cache_dir}/playground_bead_p-2to-6.alu'
        self.dp.map_graphical_pointers()
        self.solver.particle_x.read_file(initial_particle_x_filename)
        self.solver.particle_v.read_file(initial_particle_v_filename)
        self.solver.particle_pressure.read_file(
            initial_particle_pressure_filename)
        self.dp.unmap_graphical_pointers()

        self.solver.reset_solving_var()
        self.solver.t = 0
        self.simulation_sampling = FluidSamplePellets(
            self.dp, f'{self.truth_dir}/sample-x.alu')
        self.mask = self.dp.create_coated(
            (self.simulation_sampling.num_samples), 1, np.uint32)
        self.simulation_sampling.sample_x.scale(self.unit.from_real_length(1))
        self.ground_truth_v = self.dp.create_coated_like(
            self.simulation_sampling.sample_data3)
        self.buoy_v_ma95 = np.zeros((self.num_buoys, 3), self.dp.default_dtype)
        self.buoy_v_ma80 = np.zeros((self.num_buoys, 3), self.dp.default_dtype)
        self.buoy_v_ma70 = np.zeros((self.num_buoys, 3), self.dp.default_dtype)
        self.buoy_v_ma40 = np.zeros((self.num_buoys, 3), self.dp.default_dtype)

        self.dp.map_graphical_pointers()
        self.solver.update_particle_neighbors()
        state_aggregated = self.collect_state(0)
        self.dp.unmap_graphical_pointers()
        self.episode_t = 0
        return state_aggregated

    def collect_state(self, episode_t):
        truth_pile_x, truth_pile_v, truth_pile_q, truth_pile_omega = read_pile(
            f'{self.truth_dir}/{episode_t}.pile')
        buoy_x_real = truth_pile_x[1:1 + self.num_buoys]
        self.buoy_v_real = truth_pile_v[1:1 + self.num_buoys]
        self.buoy_v_ma95 = 0.95 * self.buoy_v_real + (1 -
                                                      0.95) * self.buoy_v_ma95
        self.buoy_v_ma80 = 0.80 * self.buoy_v_real + (1 -
                                                      0.80) * self.buoy_v_ma80
        self.buoy_v_ma70 = 0.70 * self.buoy_v_real + (1 -
                                                      0.70) * self.buoy_v_ma70
        self.buoy_v_ma40 = 0.40 * self.buoy_v_real + (1 -
                                                      0.40) * self.buoy_v_ma40
        buoy_q = truth_pile_q[1:1 + self.num_buoys]
        self.coil_x_real = get_coil_x_from_com(self.dp, self.unit,
                                               self.buoy_spec, buoy_x_real,
                                               buoy_q, self.num_buoys)
        # set positions for sampling around buoys in simulation
        coil_x_np = self.unit.from_real_length(self.coil_x_real)
        self.usher_sampling.sample_x.set(coil_x_np)

        self.usher_sampling.prepare_neighbor_and_boundary(
            self.runner, self.solver)
        self.usher_sampling.sample_density(self.runner)
        self.usher_sampling.sample_velocity(self.runner, self.solver)
        # self.usher_sampling.sample_vorticity(self.runner, self.solver)
        return make_state(self.dp, self.unit, self.kinematic_viscosity_real,
                          self.buoy_v_real, self.buoy_v_ma95, self.buoy_v_ma80,
                          self.buoy_v_ma70, self.buoy_v_ma40, buoy_q,
                          self.coil_x_real, self.usher_sampling,
                          self.num_buoys)

    def calculate_reward(self, episode_id):
        self.simulation_sampling.prepare_neighbor_and_boundary(
            self.runner, self.solver)
        simulation_v_real = self.simulation_sampling.sample_velocity(
            self.runner, self.solver)
        simulation_v_real.scale(self.unit.to_real_velocity(1))

        self.ground_truth_v.read_file(f'{self.truth_dir}/v-{episode_id}.alu')
        self.mask.read_file(f'{self.truth_dir}/mask-{episode_id}.alu')
        reconstruction_error = self.runner.calculate_mse_masked(
            simulation_v_real, self.ground_truth_v, self.mask,
            self.simulation_sampling.num_samples)
        result_obj = {}
        result_obj['v_error'] = reconstruction_error

        return -reconstruction_error, result_obj

    def step(self, action_aggregated_converted):
        if np.sum(np.isnan(action_aggregated_converted)) > 0:
            print(action_aggregated_converted)
            sys.exit(0)
        truth_real_freq = 100.0
        truth_real_interval = 1.0 / truth_real_freq

        set_usher_param(self.solver.usher, self.dp, self.unit,
                        self.buoy_v_real, self.coil_x_real,
                        action_aggregated_converted, self.num_buoys)
        self.dp.map_graphical_pointers()
        target_t = self.unit.from_real_time(self.episode_t *
                                            truth_real_interval)
        while (self.solver.t < target_t):
            self.solver.step()
        new_state_aggregated = self.collect_state(self.episode_t + 1)

        # find reward
        reward, result_obj = self.calculate_reward(self.episode_t + 1)
        grid_anomaly = self.dp.coat(self.solver.grid_anomaly).get()[0]

        if self.display:
            self.solver.normalize(self.solver.particle_v,
                                  self.particle_normalized_attr, 0,
                                  self.unit.from_real_velocity(0.02))
        self.dp.unmap_graphical_pointers()
        if self.display:
            self.display_proxy.draw()
        self.episode_t += 1
        done = False
        if reward < -3 or grid_anomaly > 0:
            reward -= 3
            print(f'early termination {reward}')
            done = True
        if self.episode_t == self._max_episode_steps - 1:
            done = True
        return new_state_aggregated, reward, done, result_obj

dp = al.Depot(np.float32)

max_xoffset = 0.05
max_voffset = 0.04
max_focal_dist = 0.20
min_usher_kernel_radius = 0.01
max_usher_kernel_radius = 0.08
max_strength = 1000

agent = TD3(actor_lr=3e-4,
            critic_lr=3e-4,
            critic_weight_decay=0,
            state_dim=get_state_dim(),
            action_dim=get_action_dim(),
            expl_noise_func=GaussianNoise(),
            gamma=0.95,
            min_action=np.array([
                -max_xoffset, -max_xoffset, -max_xoffset, -max_voffset,
                -max_voffset, -max_voffset, min_usher_kernel_radius, 0, -1, -1,
                -1
            ]),
            max_action=np.array([
                +max_xoffset, +max_xoffset, +max_xoffset, +max_voffset,
                +max_voffset, +max_voffset, max_usher_kernel_radius,
                max_strength, 1, 1, 1
            ]),
            learn_after=10000,
            replay_size=36000000,
            hidden_sizes=[2048, 2048, 1024],
            actor_final_scale=0.05 / np.sqrt(1000),
            critic_final_scale=1,
            soft_update_rate=0.005,
            batch_size=256)

def eval_agent(dp, agent, report_state_action=False):
    val_dirs = [
        f"{args.truth_dir}/rltruth-be268318-0526.07.32.30/",
        f"{args.truth_dir}/rltruth-5caefe43-0526.14.46.12/",
        f"{args.truth_dir}/rltruth-e8edf09d-0526.18.34.19/",
        f"{args.truth_dir}/rltruth-6de1d91b-0526.09.31.47/",
        f"{args.truth_dir}/rltruth-3b860b54-0526.23.12.15/",
        f"{args.truth_dir}/rltruth-eb3494c1-0527.00.32.34/",
        f"{args.truth_dir}/rltruth-e9ba71d8-0527.01.52.31/"
    ]
    eval_env = Environment(dp, truth_dirs=val_dirs, display=False)
    timestamp_str, timestamp_hash = get_timestamp_and_hash()

    avg_reward = 0.
    for _ in range(len(val_dirs)):
        state, done = eval_env.reset(), False
        if (report_state_action):
            state_history = np.zeros((eval_env._max_episode_steps, eval_env.num_buoys, get_state_dim()))
            action_history = np.zeros((eval_env._max_episode_steps, eval_env.num_buoys, get_action_dim()))
            cursor = 0
        while not done:
            action = agent.get_action(state, enable_noise=False)
            real_action = agent.actor.from_normalized_action(action)
            if (report_state_action):
                state_history[cursor] = state
                action_history[cursor] = real_action
                cursor+=1
            state, reward, done, info = eval_env.step(real_action)
            avg_reward += reward
        if (report_state_action):
            report_save_dir = Path(f'val-{Path(eval_env.truth_dir).name}-{timestamp_hash}')
            report_save_dir.mkdir(parents=True)
            np.save(f'{str(report_save_dir)}/state.npy', state_history)
            np.save(f'{str(report_save_dir)}/action.npy', action_history)

    avg_reward /= len(val_dirs)


    return avg_reward


agent.load_models(args.model_dir)
eval_agent(dp, agent, report_state_action=True)
