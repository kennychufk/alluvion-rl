import sys
import math
import random
import alluvion as al
import numpy as np
import torch

from .unit import Unit
from .fluid_sample_pellets import FluidSamplePellets
from .fluid_sample import FluidSample
from .parameterize_viscosity import parameterize_kinematic_viscosity_with_pellets, parameterize_kinematic_viscosity
from .policy_codec import make_state, set_usher_param, get_coil_x_from_com
from .buoy_spec import BuoySpec
from .io import read_pile
from .rigid_interpolator import BuoyInterpolator


class Environment:

    def get_num_buoys(self, truth_dir):
        return self.dp.Pile.get_size_from_file(f'{truth_dir}/0.pile') - 2

    def find_max_num_buoys(self):
        max_num_buoys = 0
        for truth_dir in self.truth_dirs:
            num_buoys = self.get_num_buoys(truth_dir)
            if num_buoys > max_num_buoys:
                max_num_buoys = num_buoys
        return max_num_buoys

    def find_max_num_beads(self):
        max_num_beads = 0
        for truth_dir in self.truth_dirs:
            unit = Unit(
                real_kernel_radius=self.unit.rl,
                real_density0=np.load(f'{truth_dir}/density0_real.npy').item(),
                real_gravity=self.unit.rg)
            fluid_mass = unit.from_real_mass(
                np.load(f'{truth_dir}/fluid_mass.npy').item())
            num_beads = int(fluid_mass / self.cn.particle_mass)
            if num_beads > max_num_beads:
                max_num_beads = num_beads
        return max_num_beads

    def get_truth_num_beads(self, truth_dir):
        return self.dp.get_alu_info(f'{truth_dir}/v2-0.alu')[0][0]

    def find_truth_max_num_beads(self):
        max_num_beads = 0
        for truth_dir in self.truth_dirs:
            num_beads = self.get_truth_num_beads(truth_dir)
            if num_beads > max_num_beads:
                max_num_beads = num_beads
        return max_num_beads

    def get_simulation_sampling(self, truth_dir):
        simulation_sampling = FluidSamplePellets(
            self.dp, f'{truth_dir}/sample-x.alu', self.cni
        ) if self.volume_method == al.VolumeMethod.pellets else FluidSample(
            self.dp, f'{truth_dir}/sample-x.alu')
        simulation_sampling.sample_x.scale(self.unit.from_real_length(1))
        return simulation_sampling

    def init_real_kernel_radius(self):
        self.real_kernel_radius = 2**-6

    def init_container_pellet_file(self):
        self.container_pellet_filename = f'{self.cache_dir}/cube24-{self.real_kernel_radius}.alu'

    def __init__(self,
                 dp,
                 truth_dirs,
                 cache_dir,
                 ma_alphas,
                 display,
                 volume_method=al.VolumeMethod.pellets,
                 save_visual=False,
                 reward_option=0):
        self.dp = dp
        self.cn, self.cni = self.dp.create_cn()
        self.display = display
        if display and not self.dp.has_display():
            self.dp.create_display(800, 600, "", False)
        self.display_proxy = self.dp.get_display_proxy() if display else None
        self.runner = self.dp.Runner()
        self.volume_method = volume_method
        self.cache_dir = cache_dir
        self.save_visual = save_visual
        self.reward_option = reward_option

        # === constants
        self.init_real_kernel_radius()
        self.init_container_pellet_file()
        particle_radius = 0.25
        kernel_radius = 1.0
        density0 = 1.0
        cubical_particle_volume = 8 * particle_radius * particle_radius * particle_radius
        volume_relative_to_cube = 0.8
        particle_mass = cubical_particle_volume * volume_relative_to_cube * density0

        gravity = self.dp.f3(0, -1, 0)
        self.unit = Unit(
            real_kernel_radius=self.real_kernel_radius,
            real_density0=1,  # dummy
            real_gravity=-9.80665)

        self.cn.set_kernel_radius(kernel_radius)
        self.cn.set_particle_attr(particle_radius, particle_mass, density0)
        self.cn.boundary_epsilon = 1e-9
        self.cn.gravity = gravity
        self.truth_dirs = truth_dirs
        self.ma_alphas = ma_alphas
        # === constants

        max_num_beads = self.find_max_num_beads()
        print('max_num_beads', max_num_beads)
        max_num_buoys = self.find_max_num_buoys()
        print('max_num_buoys', max_num_buoys)
        truth_max_num_beads = self.find_truth_max_num_beads()

        container_num_pellets = self.dp.get_alu_info(
            self.container_pellet_filename)[0][0]

        # rigids
        self.pile = self.dp.Pile(self.dp,
                                 self.runner,
                                 max_num_contacts=0,
                                 volume_method=volume_method,
                                 max_num_pellets=container_num_pellets,
                                 cn=self.cn,
                                 cni=self.cni)

        ## ================== container
        self.container_width = self.unit.from_real_length(0.24)
        container_dim = self.dp.f3(self.container_width, self.container_width,
                                   self.container_width)
        container_distance = self.dp.BoxDistance.create(container_dim,
                                                        outset=0)
        container_extent = container_distance.aabb_max - container_distance.aabb_min
        container_res_float = container_extent / particle_radius
        container_res = al.uint3(int(container_res_float.x),
                                 int(container_res_float.y),
                                 int(container_res_float.z))
        print('container_res', container_res)
        container_pellet_x = self.dp.create((container_num_pellets), 3)
        container_pellet_x.read_file(self.container_pellet_filename)
        if self.volume_method == al.VolumeMethod.pellets:
            self.pile.add_pellets(container_distance,
                                  container_res,
                                  pellets=container_pellet_x,
                                  sign=-1,
                                  mass=0,
                                  restitution=0.8,
                                  friction=0.3)
        else:
            self.pile.add(container_distance,
                          container_res,
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
        self.cni.grid_res = al.uint3(
            int(math.ceil(container_aabb_range_per_h.x)),
            int(math.ceil(container_aabb_range_per_h.y)),
            int(math.ceil(container_aabb_range_per_h.z))) + 8
        self.cni.grid_offset = al.int3(
            int(container_distance.aabb_min.x) - 4,
            int(container_distance.aabb_min.y) - 4,
            int(container_distance.aabb_min.z) - 4)
        self.cni.max_num_particles_per_cell = 64
        self.cni.max_num_neighbors_per_particle = 64

        self._max_episode_steps = 1000
        self._reward_delay = 0
        self.solver = self.dp.SolverI(self.runner,
                                      self.pile,
                                      self.dp,
                                      max_num_particles=max_num_beads,
                                      num_ushers=max_num_buoys,
                                      enable_surface_tension=False,
                                      enable_vorticity=False,
                                      cn=self.cn,
                                      cni=self.cni,
                                      graphical=display)
        self.solver.max_dt = self.unit.from_real_time(0.0005)
        self.solver.initial_dt = self.solver.max_dt
        self.solver.min_dt = 0
        self.solver.cfl = 0.2
        self.usher_sampling = FluidSamplePellets(
            self.dp, np.zeros(
                (max_num_buoys, 3), self.dp.default_dtype), self.cni
        ) if self.volume_method == al.VolumeMethod.pellets else FluidSample(
            self.dp, np.zeros((max_num_buoys, 3), self.dp.default_dtype))
        self.dir_id = 0
        self.simulation_sampling = None
        self.ground_truth_v = None
        self.v_zero = None
        self.weight = None
        # KL Divergence
        self.partial_histogram = self.dp.create_coated(
            (al.kPartialHistogram256Size), 1, np.uint32)
        self.histogram_sim = self.dp.create_coated((al.kHistogram256BinCount),
                                                   1, np.uint32)
        self.histogram_truth = self.dp.create_coated(
            (al.kHistogram256BinCount), 1, np.uint32)
        self.bead_v = self.dp.create_coated((self.solver.max_num_particles), 3)
        self.bead_v_component = self.dp.create_coated(
            (self.solver.max_num_particles), 1)
        self.particle_v_truth = self.dp.create_coated((truth_max_num_beads), 1)
        self.quantized4s_sim = dp.create_coated(
            ((self.solver.max_num_particles - 1) // 4 + 1), 1, np.uint32)
        self.quantized4s_truth = dp.create_coated(
            ((truth_max_num_beads - 1) // 4 + 1), 1, np.uint32)

        self.truth_real_freq = 100.0
        self.truth_real_interval = 1.0 / self.truth_real_freq

        self.visual_real_interval = 1.0 / 100.0
        self.next_visual_frame_id = 0
        if self.save_visual:
            self.visual_x_scaled = self.dp.create_coated_like(
                self.solver.particle_x)
            self.visual_v2_scaled = self.dp.create_coated_like(
                self.solver.particle_cfl_v2)
            self.save_dir_visual = None

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
        np.random.seed(seed_num)
        random.seed(seed_num)
        torch.manual_seed(seed_num)

    def reset_truth_dir(self):
        self.truth_dir = self.truth_dirs[self.dir_id]
        self.dir_id = (self.dir_id + 1) % len(self.truth_dirs)
        print(self.dir_id, self.truth_dir)

    def reset_vectors(self):
        self.num_buoys = self.get_num_buoys(self.truth_dir)
        self.truth_num_beads = self.get_truth_num_beads(self.truth_dir)
        self.buoy_v_ma0 = np.zeros((self.num_buoys, 3), self.dp.default_dtype)
        self.buoy_v_ma1 = np.zeros((self.num_buoys, 3), self.dp.default_dtype)
        self.buoy_v_ma2 = np.zeros((self.num_buoys, 3), self.dp.default_dtype)
        self.buoy_v_ma3 = np.zeros((self.num_buoys, 3), self.dp.default_dtype)
        if self.simulation_sampling is not None:
            self.simulation_sampling.destroy_variables()
        if self.ground_truth_v is not None:
            self.dp.remove(self.ground_truth_v)
        if self.v_zero is not None:
            self.dp.remove(self.v_zero)
        if self.weight is not None:
            self.dp.remove(self.weight)
        self.simulation_sampling = self.get_simulation_sampling(self.truth_dir)
        self.ground_truth_v = self.dp.create_coated_like(
            self.simulation_sampling.sample_data3)
        self.v_zero = self.dp.create_coated_like(
            self.simulation_sampling.sample_data3)
        self.v_zero.set_zero()
        self.weight = self.dp.create_coated(
            (self.simulation_sampling.num_samples), 1)
        # self.histogram_x_all = np.load(f'{self.truth_dir}/vx-hist.npy')
        # self.histogram_y_all = np.load(f'{self.truth_dir}/vy-hist.npy')
        # self.histogram_z_all = np.load(f'{self.truth_dir}/vz-hist.npy')

    def reset_buoy_interpolators(self):
        marker_dtype = np.dtype([('t', np.float32), ('x', np.float32, 3),
                                 ('q', np.float32, 4)])
        buoy_trajectories = np.empty((self.num_buoys, self._max_episode_steps),
                                     dtype=marker_dtype)
        for episode_t in range(self._max_episode_steps):
            truth_pile_x, truth_pile_v, truth_pile_q, truth_pile_omega = read_pile(
                f'{self.truth_dir}/{episode_t}.pile')
            buoy_trajectories[:, episode_t]['x'] = truth_pile_x[1:1 +
                                                                self.num_buoys]
            buoy_trajectories[:, episode_t]['q'] = truth_pile_q[1:1 +
                                                                self.num_buoys]
        for buoy_id in range(self.num_buoys):
            buoy_trajectories[buoy_id]['x'] += np.tile(
                np.random.normal(scale=2e-3, size=(3)),
                self._max_episode_steps).reshape(-1, 3)
        self.buoy_interpolators = [
            BuoyInterpolator(self.dp,
                             sample_interval=0.01,
                             trajectory=trajectory)
            for trajectory in buoy_trajectories
        ]

    def reset_solver_properties(self):
        self.real_density0 = np.load(f'{self.truth_dir}/density0_real.npy')
        self.unit = Unit(real_kernel_radius=self.unit.rl,
                         real_density0=self.real_density0,
                         real_gravity=self.unit.rg)
        self.kinematic_viscosity_real = np.load(
            f'{self.truth_dir}/kinematic_viscosity_real.npy').item()

        self.cn.set_particle_attr(self.cn.particle_radius,
                                  self.cn.particle_mass, self.cn.density0)
        if self.volume_method == al.VolumeMethod.pellets:
            self.cn.viscosity, self.cn.boundary_viscosity = self.unit.from_real_kinematic_viscosity(
                parameterize_kinematic_viscosity_with_pellets(
                    self.kinematic_viscosity_real))
        else:
            self.cn.viscosity, self.cn.boundary_viscosity = self.unit.from_real_kinematic_viscosity(
                parameterize_kinematic_viscosity(
                    self.kinematic_viscosity_real))

        self.real_fluid_mass = np.load(
            f'{self.truth_dir}/fluid_mass.npy').item()
        self.fluid_mass = self.unit.from_real_mass(self.real_fluid_mass)
        num_particles = int(self.fluid_mass / self.cn.particle_mass)
        self.solver.num_particles = num_particles

        self.solver.usher.reset()
        self.solver.usher.num_ushers = self.num_buoys
        print('num_particles', num_particles, 'self.num_buoys', self.num_buoys)

    def reset_solver_initial(self):
        postfix = f"-{self.real_kernel_radius}-{float(self.real_density0)}-{float(self.real_fluid_mass)}.alu"
        self.initial_particle_x_filename = f'{self.cache_dir}/playground_bead_x{postfix}'
        self.initial_particle_v_filename = f'{self.cache_dir}/playground_bead_v{postfix}'
        self.initial_particle_pressure_filename = f'{self.cache_dir}/playground_bead_p{postfix}'

        self.dp.map_graphical_pointers()
        print(self.initial_particle_x_filename)
        self.solver.particle_x.read_file(self.initial_particle_x_filename)
        self.solver.particle_v.read_file(self.initial_particle_v_filename)
        self.solver.particle_pressure.read_file(
            self.initial_particle_pressure_filename)
        self.dp.unmap_graphical_pointers()

        self.solver.reset_solving_var()
        self.solver.t = 0

    def reset(self):
        self.reset_truth_dir()
        self.reset_vectors()
        self.reset_buoy_interpolators()
        self.reset_solver_properties()
        self.reset_solver_initial()
        self.dp.copy_cn_external(self.cn, self.cni)

        self.dp.map_graphical_pointers()
        state_aggregated = self.collect_state(0)
        self.dp.unmap_graphical_pointers()
        self.episode_t = 0
        return state_aggregated

    def get_buoy_kinematics_real(self, episode_t):
        t_real = episode_t * self.truth_real_interval
        buoys_x = np.zeros((self.num_buoys, 3), self.dp.default_dtype)
        buoys_v = np.zeros((self.num_buoys, 3), self.dp.default_dtype)
        buoys_q = np.zeros((self.num_buoys, 4), self.dp.default_dtype)

        for buoy_id in range(self.num_buoys):
            buoys_x[buoy_id] = self.buoy_interpolators[buoy_id].get_x(t_real)
            buoys_v[buoy_id] = self.buoy_interpolators[buoy_id].get_v(t_real)
            buoys_q[buoy_id] = self.buoy_interpolators[buoy_id].get_q(t_real)
        return buoys_x, buoys_v, buoys_q

    def collect_state(self, episode_t):
        buoy_x_real, self.buoy_v_real, buoy_q = self.get_buoy_kinematics_real(
            episode_t)
        self.buoy_v_ma0 = self.ma_alphas[0] * self.buoy_v_real + (
            1 - self.ma_alphas[0]) * self.buoy_v_ma0
        self.buoy_v_ma1 = self.ma_alphas[1] * self.buoy_v_real + (
            1 - self.ma_alphas[1]) * self.buoy_v_ma1
        self.buoy_v_ma2 = self.ma_alphas[2] * self.buoy_v_real + (
            1 - self.ma_alphas[2]) * self.buoy_v_ma2
        self.buoy_v_ma3 = self.ma_alphas[3] * self.buoy_v_real + (
            1 - self.ma_alphas[3]) * self.buoy_v_ma3
        self.coil_x_real = get_coil_x_from_com(self.dp, self.unit,
                                               self.buoy_spec, buoy_x_real,
                                               buoy_q, self.num_buoys)
        # set positions for sampling around buoys in simulation
        coil_x_np = self.unit.from_real_length(self.coil_x_real)
        self.usher_sampling.sample_x.set(coil_x_np)

        self.dp.copy_cn_external(self.cn, self.cni)
        self.usher_sampling.prepare_neighbor_and_boundary(
            self.runner, self.solver)
        self.usher_sampling.sample_density(self.runner)
        self.usher_sampling.sample_velocity(self.runner, self.solver)
        # self.usher_sampling.sample_vorticity(self.runner, self.solver)
        return make_state(self.dp, self.unit, self.kinematic_viscosity_real,
                          self.buoy_v_real, self.buoy_v_ma0, self.buoy_v_ma1,
                          self.buoy_v_ma2, self.buoy_v_ma3, buoy_q,
                          self.coil_x_real, self.usher_sampling,
                          self.num_buoys)

    def calculate_reward(self, episode_t):
        if episode_t - self._reward_delay < 0:
            result_obj = {}
            result_obj['v_error'] = 0
            result_obj['truth_sqr'] = 0
            result_obj['num_masked'] = 0
            return 0, result_obj
        if self.reward_option == 0:
            self.simulation_sampling.prepare_neighbor_and_boundary(
                self.runner, self.solver)
            simulation_v_real = self.simulation_sampling.sample_velocity(
                self.runner, self.solver)
            simulation_v_real.scale(self.unit.to_real_velocity(1))

            self.ground_truth_v.read_file(
                f'{self.truth_dir}/v-{episode_t-self._reward_delay}.alu')
            # self.weight.read_file(
            #     f'{self.truth_dir}/density-weight-{episode_t-self._reward_delay}.alu'
            # )
            self.weight.fill(1)
            v_error = self.runner.calculate_se_weighted(
                simulation_v_real, self.ground_truth_v, self.weight,
                self.simulation_sampling.num_samples)
            truth_sqr = self.runner.calculate_se_weighted(
                self.v_zero, self.ground_truth_v, self.weight,
                self.simulation_sampling.num_samples)
            result_obj = {}
            result_obj['v_error'] = v_error
            result_obj['truth_sqr'] = truth_sqr
            result_obj['num_masked'] = self.runner.sum(
                self.weight, self.simulation_sampling.num_samples)
            return -v_error / result_obj['num_masked'], result_obj
        else:
            # KL divergence
            self.bead_v_component.set_from(self.solver.particle_cfl_v2,
                                           self.solver.num_particles)
            self.runner.sqrt_inplace(self.bead_v_component,
                                     self.solver.num_particles)
            self.bead_v_component.scale(self.unit.to_real_velocity(1))
            max_v_bin = 0.25
            self.runner.launch_histogram256(self.partial_histogram,
                                            self.histogram_sim,
                                            self.quantized4s_sim,
                                            self.bead_v_component, 0,
                                            max_v_bin,
                                            self.solver.num_particles)
            # self.bead_v.set_from(self.solver.particle_v,
            #                      self.solver.num_particles)
            # self.bead_v.scale(self.unit.to_real_velocity(1))

            # self.runner.extract_x(self.bead_v, self.bead_v_component,
            #                       self.solver.num_particles)
            # self.runner.launch_histogram256(self.partial_histogram,
            #                                 self.histogram_sim,
            #                                 self.quantized4s_sim,
            #                                 self.bead_v_component, -max_v_bin,
            #                                 max_v_bin,
            #                                 self.solver.num_particles)
            # self.histogram_truth.set(self.histogram_x_all[episode_t -
            #                                               self._reward_delay])
            # kl_div_x = self.runner.calculate_kl_divergence(
            #     self.histogram_sim, self.histogram_truth,
            #     self.solver.num_particles, self.truth_num_beads)

            # self.runner.extract_y(self.bead_v, self.bead_v_component,
            #                       self.solver.num_particles)
            # self.runner.launch_histogram256(self.partial_histogram,
            #                                 self.histogram_sim,
            #                                 self.quantized4s_sim,
            #                                 self.bead_v_component, -max_v_bin,
            #                                 max_v_bin,
            #                                 self.solver.num_particles)
            # self.histogram_truth.set(self.histogram_y_all[episode_t -
            #                                               self._reward_delay])
            # kl_div_y = self.runner.calculate_kl_divergence(
            #     self.histogram_sim, self.histogram_truth,
            #     self.solver.num_particles, self.truth_num_beads)

            # self.runner.extract_z(self.bead_v, self.bead_v_component,
            #                       self.solver.num_particles)
            # self.runner.launch_histogram256(self.partial_histogram,
            #                                 self.histogram_sim,
            #                                 self.quantized4s_sim,
            #                                 self.bead_v_component, -max_v_bin,
            #                                 max_v_bin,
            #                                 self.solver.num_particles)
            # self.histogram_truth.set(self.histogram_z_all[episode_t -
            #                                               self._reward_delay])
            # kl_div_z = self.runner.calculate_kl_divergence(
            #     self.histogram_sim, self.histogram_truth,
            #     self.solver.num_particles, self.truth_num_beads)

            # kl_div = kl_div_x + kl_div_y + kl_div_z
            self.particle_v_truth.read_file(
                f'{self.truth_dir}/v2-{episode_t-self._reward_delay}.alu')
            self.runner.sqrt_inplace(self.particle_v_truth,
                                     self.truth_num_beads)
            self.runner.launch_histogram256(self.partial_histogram,
                                            self.histogram_truth,
                                            self.quantized4s_truth,
                                            self.particle_v_truth, 0,
                                            max_v_bin, self.truth_num_beads)
            kl_div = self.runner.calculate_kl_divergence(
                self.histogram_sim, self.histogram_truth,
                self.solver.num_particles, self.truth_num_beads)
            result_obj = {}
            result_obj['v_error'] = kl_div
            result_obj['truth_sqr'] = 0
            result_obj['num_masked'] = 0
            return -kl_div, result_obj

    def step(self, action_aggregated_converted):
        if np.sum(np.isnan(action_aggregated_converted)) > 0:
            print(action_aggregated_converted)
            sys.exit(0)

        set_usher_param(self.solver.usher, self.dp, self.unit,
                        self.buoy_v_real, self.coil_x_real,
                        action_aggregated_converted, self.num_buoys)
        self.dp.map_graphical_pointers()
        target_t = self.unit.from_real_time(self.episode_t *
                                            self.truth_real_interval)
        while (self.solver.t < target_t):
            self.solver.step()
            if self.save_visual and self.solver.t >= self.unit.from_real_time(
                    self.next_visual_frame_id * self.visual_real_interval):
                self.visual_v2_scaled.set_from(self.solver.particle_cfl_v2)
                self.visual_v2_scaled.scale(self.unit.to_real_velocity_mse(1))
                self.visual_x_scaled.set_from(self.solver.particle_x,
                                              self.solver.num_particles)
                self.visual_x_scaled.scale(self.unit.to_real_length(1))
                if self.save_dir_visual is not None:
                    self.visual_v2_scaled.write_file(
                        f'{str(self.save_dir_visual)}/v2-{self.next_visual_frame_id}.alu',
                        self.solver.num_particles)
                    self.visual_x_scaled.write_file(
                        f'{str(self.save_dir_visual)}/x-{self.next_visual_frame_id}.alu',
                        self.solver.num_particles)
                    self.pile.write_file(
                        f'{str(self.save_dir_visual)}/{self.next_visual_frame_id}.pile',
                        self.unit.to_real_length(1),
                        self.unit.to_real_velocity(1),
                        self.unit.to_real_angular_velocity(1))
                self.next_visual_frame_id += 1

        new_state_aggregated = self.collect_state(self.episode_t + 1)

        # find reward
        reward, result_obj = self.calculate_reward(self.episode_t + 1)
        grid_anomaly = self.dp.coat(
            self.solver.grid_anomaly).get()[0]  # TODO: use sum

        if self.display:
            self.solver.normalize(self.solver.particle_v,
                                  self.particle_normalized_attr, 0,
                                  self.unit.from_real_velocity(0.02))
        self.dp.unmap_graphical_pointers()
        if self.display:
            self.display_proxy.draw()
        self.episode_t += 1
        done = False
        if reward < -1000 or grid_anomaly > 0:
            print(f'early termination {reward} grid_anomaly {grid_anomaly}')
            reward -= 3
            done = True
        if self.episode_t == self._max_episode_steps - 1:
            done = True
        return new_state_aggregated, reward, done, result_obj


class EnvironmentPIV(Environment):

    def init_real_kernel_radius(self):
        self.real_kernel_radius = 0.011

    def init_container_pellet_file(self):
        self.container_pellet_filename = f'{self.cache_dir}/cube24-0.011.alu'

    def get_num_buoys(self, truth_dir):
        return len(np.load(f'{truth_dir}/rec/marker_ids.npy'))

    def find_truth_max_num_beads(self):
        return 0

    def get_truth_num_beads(self, truth_dir):
        unit = Unit(real_kernel_radius=self.unit.rl,
                    real_density0=np.load(f'{truth_dir}/density0_real.npy'),
                    real_gravity=self.unit.rg)
        num_beads = int(
            np.load(f'{truth_dir}/fluid_mass.npy').item() /
            unit.to_real_mass(self.cn.particle_mass))
        return num_beads

    def __init__(self,
                 dp,
                 truth_dirs,
                 cache_dir,
                 ma_alphas,
                 display,
                 buoy_filter_postfix='-f18',
                 volume_method=al.VolumeMethod.pellets,
                 save_visual=False):
        super().__init__(dp, truth_dirs, cache_dir, ma_alphas, display,
                         volume_method, save_visual)
        self.container_shift = dp.f3(0, self.container_width * 0.5, 0)
        self.pile.x[0] = self.container_shift
        self.cni.grid_offset.y = -4
        self.dp.copy_cn()
        self.buoy_filter_postfix = buoy_filter_postfix
        self.truth_real_freq = 500.0
        self.truth_real_interval = 1.0 / self.truth_real_freq

    def get_simulation_sampling(self, truth_dir):
        sample_x_piv = np.load(f'{truth_dir}/mat_results/pos.npy').reshape(
            -1, 2)
        sample_x_np = np.zeros((len(sample_x_piv), 3),
                               dtype=self.dp.default_dtype)
        sample_x_np[:, 2] = sample_x_piv[:, 0]
        sample_x_np[:, 1] = sample_x_piv[:, 1]

        return FluidSamplePellets(
            self.dp, self.unit.from_real_length(sample_x_np), self.cni
        ) if self.volume_method == al.VolumeMethod.pellets else FluidSample(
            self.dp, self.unit.from_real_length(sample_x_np))

    def reset_vectors(self):
        super().reset_vectors()
        self.mask_collection = np.load(
            f'{self.truth_dir}/mat_results/mask-at0.0424264.npy').reshape(
                -1, self.simulation_sampling.num_samples).astype(np.float32)

        truth_v_piv = np.load(
            f'{self.truth_dir}/mat_results/vel_original.npy').reshape(
                -1, self.simulation_sampling.num_samples, 2)
        self.truth_v_collection = np.zeros((*truth_v_piv.shape[:-1], 3))
        self.truth_v_collection[..., 2] = truth_v_piv[..., 0]
        self.truth_v_collection[..., 1] = truth_v_piv[..., 1]
        self._max_episode_steps = len(truth_v_piv)

    def reset_buoy_interpolators(self):
        used_buoy_ids = np.load(f'{self.truth_dir}/rec/marker_ids.npy')
        buoy_trajectories = [
            np.load(
                f'{self.truth_dir}/rec/marker-{used_buoy_id}{self.buoy_filter_postfix}.npy'
            ) for used_buoy_id in used_buoy_ids
        ]
        self.buoy_interpolators = [
            BuoyInterpolator(self.dp,
                             sample_interval=0.01,
                             trajectory=trajectory)
            for trajectory in buoy_trajectories
        ]

    def reset_solver_initial(self):
        super().reset_solver_initial()
        self.dp.map_graphical_pointers()
        self.solver.particle_x.shift(self.container_shift,
                                     self.solver.num_particles)
        self.dp.unmap_graphical_pointers()

    def get_buoy_kinematics_real(self, episode_t):
        t_real = episode_t * self.truth_real_interval
        buoy_x_shift = np.array(
            [0, -0.026, 0], self.dp.default_dtype
        )  # coil center to board surface: 21mm, tank thickness: 5mm
        buoys_x = np.zeros((self.num_buoys, 3), self.dp.default_dtype)
        buoys_v = np.zeros((self.num_buoys, 3), self.dp.default_dtype)
        buoys_q = np.zeros((self.num_buoys, 4), self.dp.default_dtype)

        for buoy_id in range(self.num_buoys):
            buoys_x[buoy_id] = self.buoy_interpolators[buoy_id].get_x(
                t_real) + buoy_x_shift
            buoys_v[buoy_id] = self.buoy_interpolators[buoy_id].get_v(t_real)
            buoys_q[buoy_id] = self.buoy_interpolators[buoy_id].get_q(t_real)
        return buoys_x, buoys_v, buoys_q

    def calculate_reward(self, episode_t):
        if episode_t - self._reward_delay < 0:
            result_obj = {}
            result_obj['v_error'] = 0
            result_obj['truth_sqr'] = 0
            result_obj['num_masked'] = 0
            return 0, result_obj
        self.simulation_sampling.prepare_neighbor_and_boundary(
            self.runner, self.solver)
        simulation_v_real = self.simulation_sampling.sample_velocity(
            self.runner, self.solver)
        simulation_v_real.scale(self.unit.to_real_velocity(1))

        self.ground_truth_v.set(self.truth_v_collection[episode_t -
                                                        self._reward_delay])
        self.weight.set(self.mask_collection[episode_t - self._reward_delay])
        v_error = self.runner.calculate_se_yz_masked(
            simulation_v_real, self.ground_truth_v, self.weight,
            self.simulation_sampling.num_samples)
        truth_sqr = self.runner.calculate_se_yz_masked(
            self.v_zero, self.ground_truth_v, self.weight,
            self.simulation_sampling.num_samples)
        result_obj = {}
        result_obj['v_error'] = v_error
        result_obj['truth_sqr'] = truth_sqr
        result_obj['num_masked'] = np.sum(
            self.mask_collection[episode_t - self._reward_delay])

        return -v_error, result_obj
