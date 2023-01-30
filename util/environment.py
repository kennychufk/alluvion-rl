import sys
import math
import random
import alluvion as al
import alluvol
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
from . import read_alu


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
        return np.load(f'{truth_dir}/num_particles.npy').item()

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
                 reward_metric='eulerian',
                 evaluation_metrics=None,
                 shape_dir=None,
                 quick_mode=True):
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
        self.quick_mode = quick_mode

        self.reward_metric = reward_metric
        self.metrics = {}
        if reward_metric is not None:
            self.metrics[reward_metric] = 1
        if evaluation_metrics is not None:
            for metric in evaluation_metrics:
                if metric != reward_metric:
                    self.metrics[metric] = 0

        # === constants
        self.real_kernel_radius = 2**-6 if quick_mode else 0.011
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
        self.max_num_buoys = max_num_buoys
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
        if "statistical" in self.metrics and shape_dir is None:
            raise Exception(
                "Statistical metric is needed but shape_dir is unspecified")
        self.shape_dir = shape_dir
        self.recon_raster_radius = self.unit.rl * 0.36
        self.recon_voxel_size = self.recon_raster_radius / np.sqrt(3.0)

        ## ================== container
        self.container_width_real = 0.24
        self.container_width = self.unit.from_real_length(
            self.container_width_real)
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
        if quick_mode:
            self.solver.max_dt = self.unit.from_real_time(0.0005)
            self.solver.cfl = 0.2
        else:
            self.solver.max_dt = self.unit.from_real_time(0.0004)
            self.solver.cfl = 0.16
        self.solver.initial_dt = self.solver.max_dt
        self.solver.min_dt = 0
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
        self.squared_error = None
        self.buoy_reward_mask = None
        self.buoy_x = self.dp.create_coated((self.max_num_buoys), 3)
        self.particle_guiding_norm = self.dp.create_coated(
            (self.solver.max_num_particles), 1)

        if 'statistical' in self.metrics:
            self.partial_histogram = self.dp.create_coated(
                (al.kPartialHistogram256Size), 1, np.uint32)
            self.histogram_baseline = self.dp.create_coated(
                (al.kHistogram256BinCount), 1, np.uint32)
            self.histogram_sim = self.dp.create_coated(
                (al.kHistogram256BinCount), 1, np.uint32)
            self.histogram_truth = self.dp.create_coated(
                (al.kHistogram256BinCount), 1, np.uint32)
            self.bead_v_sim = self.dp.create_coated(
                (self.solver.max_num_particles), 1)
            self.bead_v_truth = self.dp.create_coated((truth_max_num_beads), 1)
            self.quantized4s_sim = dp.create_coated(
                ((self.solver.max_num_particles - 1) // 4 + 1), 1, np.uint32)
            self.quantized4s_truth = dp.create_coated(
                ((truth_max_num_beads - 1) // 4 + 1), 1, np.uint32)
            self.max_v_bin = 0.25
        if 'potential_energy' in self.metrics:
            self.bead_y_sim = self.dp.create_coated(
                (self.solver.max_num_particles), 1)
            self.bead_y_truth = self.dp.create_coated((truth_max_num_beads), 1)
            self.bead_truth = self.dp.create_coated((truth_max_num_beads), 3)

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
        if self.squared_error is not None:
            self.dp.remove(self.squared_error)
        if self.buoy_reward_mask is not None:
            self.dp.remove(self.buoy_reward_mask)
        self.simulation_sampling = self.get_simulation_sampling(self.truth_dir)
        self.ground_truth_v = self.dp.create_coated_like(
            self.simulation_sampling.sample_data3)
        self.v_zero = self.dp.create_coated_like(
            self.simulation_sampling.sample_data3)
        self.v_zero.set_zero()
        self.weight = self.dp.create_coated(
            (self.simulation_sampling.num_samples), 1)
        self.squared_error = self.dp.create_coated(
            (self.simulation_sampling.num_samples), 1)
        self.buoy_reward_mask = self.dp.create_coated(
            (self.max_num_buoys, self.simulation_sampling.num_samples), 1)
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

    def reset_volumetric_height_field(self):
        evalute_height_field = 'height_field' in self.metrics
        evalute_volumetric = 'volumetric' in self.metrics
        if evalute_volumetric or evalute_height_field:
            agitator_option = np.load(
                f'{self.truth_dir}/agitator_option.npy').item()
            self.agitator_ls = alluvol.create_mesh_level_set(
                f'{self.shape_dir}/{agitator_option}/models/manifold2-decimate-pa-dilate.obj',
                self.recon_voxel_size)
            baseline_x = self.unit.to_real_length(
                self.dp.coat(self.solver.particle_x).get(
                    self.solver.num_particles))
            self.baseline_ls = alluvol.create_liquid_level_set(
                baseline_x, self.recon_raster_radius, self.recon_voxel_size)
            if evalute_height_field:
                self.sample_layer_x = np.copy(
                    read_alu(f'{self.truth_dir}/sample-x.alu'))
                self.sample_layer_x[:, 1] = 10.0
                self.baseline_ls.setGridClassAsLevelSet()
                self.baseline_hf = alluvol.LevelSetRayIntersector(
                    self.baseline_ls).probe_heights(
                        self.sample_layer_x,
                        default_value=-self.container_width_real * 0.5)

    def reset_statistical(self):
        if 'statistical' in self.metrics:
            self.runner.norm(self.solver.particle_v, self.bead_v_sim,
                             self.solver.num_particles)
            self.bead_v_sim.scale(self.unit.to_real_velocity(1))
            self.runner.launch_histogram256(self.partial_histogram,
                                            self.histogram_baseline,
                                            self.quantized4s_sim,
                                            self.bead_v_sim, 0, self.max_v_bin,
                                            self.solver.num_particles)

    def reset_potential_energy(self):
        if 'potential_energy' in self.metrics:
            self.runner.extract_y(self.solver.particle_x, self.bead_y_sim,
                                  self.solver.num_particles)
            self.bead_y_sim.scale(self.unit.to_real_length(1))
            y_sum_sim = self.runner.sum(self.bead_y_sim,
                                        self.solver.num_particles)
            self.mean_y_sim = y_sum_sim / self.solver.num_particles
            y_diff_sim = self.bead_y_sim.get(
                self.solver.num_particles) - self.mean_y_sim
            self.pe_baseline = np.mean(y_diff_sim * y_diff_sim)

            self.bead_truth.read_file(f'{self.truth_dir}/x-0.alu')
            self.runner.extract_y(self.bead_truth, self.bead_y_truth,
                                  self.truth_num_beads)
            y_sum_truth = self.runner.sum(
                self.bead_y_truth, self.truth_num_beads)  # real unit already
            self.mean_y_truth = y_sum_truth / self.truth_num_beads

    def reset_solver_properties(self):
        self.real_density0 = np.load(f'{self.truth_dir}/density0_real.npy')
        self.unit = Unit(real_kernel_radius=self.unit.rl,
                         real_density0=self.real_density0,
                         real_gravity=self.unit.rg)
        self.truth_unit = Unit(real_kernel_radius=2**-8,
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
        self.next_visual_frame_id = 0

    def reset(self):
        self.reset_truth_dir()
        self.reset_vectors()
        self.reset_buoy_interpolators()
        self.reset_solver_properties()
        self.reset_solver_initial()
        # reward/score related
        self.reset_volumetric_height_field()
        self.reset_statistical()
        self.reset_potential_energy()
        self.dp.copy_cn_external(self.cn, self.cni)

        self.dp.map_graphical_pointers()
        state = self.collect_state(0)
        self.dp.unmap_graphical_pointers()
        self.episode_t = 0
        return state

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

        # for local reward
        self.buoy_x.set(self.unit.from_real_length(buoy_x_real))
        # self.usher_sampling.sample_vorticity(self.runner, self.solver)
        return make_state(self.dp, self.unit, self.kinematic_viscosity_real,
                          self.buoy_v_real, self.buoy_v_ma0, self.buoy_v_ma1,
                          self.buoy_v_ma2, self.buoy_v_ma3, buoy_q,
                          self.coil_x_real, self.usher_sampling,
                          self.num_buoys)

    def calculate_volumetric_error(self,
                                   episode_t,
                                   compute_local_errors=False):
        local_errors = None  # TODO: to be implemented
        if compute_local_errors:
            raise Exception(
                "Local rewards are unsupported for volumetric error")
        truth_vdb_filename = f'{self.truth_dir}/x-{episode_t}-resampled6.vdb'
        truth_ls = alluvol.FloatGrid.read(truth_vdb_filename)

        recon_x = self.unit.to_real_length(
            self.dp.coat(self.solver.particle_x).get(
                self.solver.num_particles))
        recon_ls = alluvol.create_liquid_level_set(recon_x,
                                                   self.recon_raster_radius,
                                                   self.recon_voxel_size)
        xs, vs, qs, omegas = read_pile(f'{self.truth_dir}/{episode_t}.pile')
        transformed_agitator_ls = alluvol.transform_level_set(
            self.agitator_ls, alluvol.F3(xs[-1]), alluvol.F4(qs[-1]),
            self.recon_voxel_size)

        recon_sym_diff = alluvol.csgUnionCopy(truth_ls, recon_ls)
        recon_intersection = alluvol.csgIntersectionCopy(truth_ls, recon_ls)
        alluvol.csgDifference(recon_sym_diff, recon_intersection)
        recon_error_ls = alluvol.csgDifferenceCopy(recon_sym_diff,
                                                   transformed_agitator_ls)

        baseline_sym_diff = alluvol.csgUnionCopy(truth_ls, self.baseline_ls)
        baseline_intersection = alluvol.csgIntersectionCopy(
            truth_ls, self.baseline_ls)
        alluvol.csgDifference(baseline_sym_diff, baseline_intersection)
        baseline_error_ls = alluvol.csgDifferenceCopy(baseline_sym_diff,
                                                      transformed_agitator_ls)

        if self.save_visual and self.save_dir_visual is not None:
            recon_ls.write_obj(f"{str(self.save_dir_visual)}/{episode_t}.obj")
            recon_error_ls.write_obj(
                f"{str(self.save_dir_visual)}/recon-error-{episode_t}.obj")
            baseline_error_ls.write_obj(
                f"{str(self.save_dir_visual)}/baseline-error-{episode_t}.obj")
        recon_error_ls.setGridClassAsLevelSet()
        baseline_error_ls.setGridClassAsLevelSet()
        error = recon_error_ls.calculate_volume()
        baseline = baseline_error_ls.calculate_volume()
        return error, local_errors, baseline, 1

    def calculate_height_field_error(self,
                                     episode_t,
                                     compute_local_errors=False):
        local_errors = None  # TODO: to be implemented
        if compute_local_errors:
            raise Exception(
                "Local rewards are unsupported for height field error")
        truth_vdb_filename = f'{self.truth_dir}/x-{episode_t}-resampled6.vdb'
        truth_ls = alluvol.FloatGrid.read(truth_vdb_filename)
        truth_ls.setGridClassAsLevelSet()
        truth_hf = alluvol.LevelSetRayIntersector(truth_ls).probe_heights(
            self.sample_layer_x,
            default_value=-self.container_width_real * 0.5)

        recon_x = self.unit.to_real_length(
            self.dp.coat(self.solver.particle_x).get(
                self.solver.num_particles))
        recon_ls = alluvol.create_liquid_level_set(recon_x,
                                                   self.recon_raster_radius,
                                                   self.recon_voxel_size)
        recon_ls.setGridClassAsLevelSet()
        recon_hf = alluvol.LevelSetRayIntersector(recon_ls).probe_heights(
            self.sample_layer_x,
            default_value=-self.container_width_real * 0.5)
        recon_diff = recon_hf - truth_hf

        baseline_diff = self.baseline_hf - truth_hf

        if self.save_visual and self.save_dir_visual is not None:
            np.save(f"{str(self.save_dir_visual)}/truth-hf-{episode_t}.npy",
                    truth_hf)
            np.save(f"{str(self.save_dir_visual)}/recon-hf-{episode_t}.npy",
                    recon_hf)
            np.save(
                f"{str(self.save_dir_visual)}/baseline-hf-error-{episode_t}.npy",
                baseline_diff)
        error = np.sum(recon_diff * recon_diff)
        baseline = np.sum(baseline_diff * baseline_diff)
        return error, local_errors, baseline, 1

    def calculate_eulerian_error(self,
                                 episode_t,
                                 use_mask=False,
                                 compute_local_errors=False):
        self.simulation_sampling.prepare_neighbor_and_boundary(
            self.runner, self.solver)
        simulation_v_real = self.simulation_sampling.sample_velocity(
            self.runner, self.solver)
        simulation_v_real.scale(self.unit.to_real_velocity(1))

        self.ground_truth_v.read_file(f'{self.truth_dir}/v-{episode_t}.alu')
        if use_mask:
            self.weight.read_file(
                f'{self.truth_dir}/density-weight-{episode_t}.alu')
        else:
            self.weight.fill(1)
        self.runner.calculate_se(simulation_v_real, self.ground_truth_v,
                                 self.squared_error,
                                 self.simulation_sampling.num_samples)
        self.runner.launch_compute_distance_mask_multiple(
            self.simulation_sampling.sample_x,
            self.buoy_x,
            self.buoy_reward_mask,
            distance_threshold=self.unit.from_real_length(0.05),
            num_grid_points=self.simulation_sampling.num_samples,
            num_buoys=self.num_buoys)
        local_num_samples = np.sum(self.buoy_reward_mask.get(
            [self.num_buoys, self.simulation_sampling.num_samples]),
                                   axis=1)
        local_errors = None
        if compute_local_errors:
            local_errors = np.zeros(self.num_buoys)
            for buoy_id in range(self.num_buoys):
                local_errors[
                    buoy_id] = self.runner.sum_products_different_offsets(
                        self.squared_error,
                        self.buoy_reward_mask,
                        self.simulation_sampling.num_samples,
                        offset0=0,
                        offset1=self.simulation_sampling.num_samples * buoy_id)
            local_errors /= local_num_samples  # NOTE: assume that buoys never leave the grid

        error = self.runner.sum_products(self.squared_error, self.weight,
                                         self.simulation_sampling.num_samples)
        baseline = self.runner.calculate_se_weighted(
            self.v_zero, self.ground_truth_v, self.weight,
            self.simulation_sampling.num_samples)
        num_samples = self.runner.sum(self.weight,
                                      self.simulation_sampling.num_samples)
        return error, local_errors, baseline, num_samples

    def calculate_eulerian_masked_error(self,
                                        episode_t,
                                        compute_local_errors=False):
        return self.calculate_eulerian_error(
            episode_t,
            use_mask=True,
            compute_local_errors=compute_local_errors)

    def calculate_statistical_error(self,
                                    episode_t,
                                    compute_local_errors=False):
        local_errors = None  # TODO: to be implemented
        if compute_local_errors:
            raise Exception("Local rewards are unsupported for KL Divergence")
        self.bead_v_sim.set_from(self.solver.particle_cfl_v2,
                                 self.solver.num_particles)
        self.runner.sqrt_inplace(self.bead_v_sim, self.solver.num_particles)
        self.bead_v_sim.scale(self.unit.to_real_velocity(1))
        self.runner.launch_histogram256(self.partial_histogram,
                                        self.histogram_sim,
                                        self.quantized4s_sim, self.bead_v_sim,
                                        0, self.max_v_bin,
                                        self.solver.num_particles)
        self.bead_v_truth.read_file(f'{self.truth_dir}/v2-{episode_t}.alu')
        self.runner.sqrt_inplace(self.bead_v_truth, self.truth_num_beads)
        self.runner.launch_histogram256(self.partial_histogram,
                                        self.histogram_truth,
                                        self.quantized4s_truth,
                                        self.bead_v_truth, 0, self.max_v_bin,
                                        self.truth_num_beads)
        error = self.runner.calculate_kl_divergence(self.histogram_sim,
                                                    self.histogram_truth,
                                                    self.solver.num_particles,
                                                    self.truth_num_beads)
        baseline = self.runner.calculate_kl_divergence(
            self.histogram_baseline, self.histogram_truth,
            self.solver.num_particles, self.truth_num_beads)
        return error, local_errors, baseline, 1

    def calculate_potential_energy_error(self, episode_t,
                                         compute_local_errors):
        local_errors = None
        if compute_local_errors:
            raise Exception(
                "Local rewards are unsupported for potential energy")
        self.runner.extract_y(self.solver.particle_x, self.bead_y_sim,
                              self.solver.num_particles)
        self.bead_y_sim.scale(self.unit.to_real_length(1))

        y_sum_sim = self.runner.sum(self.bead_y_sim, self.solver.num_particles)
        y_diff_sim = self.bead_y_sim.get(
            self.solver.num_particles) - self.mean_y_sim
        pe_sim = np.mean(y_diff_sim * y_diff_sim)

        self.bead_truth.read_file(f'{self.truth_dir}/x-{episode_t}.alu')
        self.runner.extract_y(self.bead_truth, self.bead_y_truth,
                              self.truth_num_beads)
        y_sum_truth = self.runner.sum(
            self.bead_y_truth, self.truth_num_beads)  # real unit already
        self.mean_y_truth = y_sum_truth / self.truth_num_beads
        y_diff_truth = self.bead_y_truth.get(
            self.truth_num_beads) - self.mean_y_truth
        # print('mean_y_sim', self.mean_y_sim, 'mean_y_truth', self.mean_y_truth)
        print(
            'pressure mean',
            self.runner.sum(self.solver.particle_pressure,
                            self.solver.num_particles) /
            self.solver.num_particles)
        pe_truth = np.mean(y_diff_truth * y_diff_truth)

        # print(pe_sim, self.pe_baseline, pe_truth)

        error = pe_sim - pe_truth
        error = error * error

        baseline = self.pe_baseline - pe_truth
        baseline = baseline * baseline

        return error, local_errors, baseline, 1

    def calculate_reward(self, episode_t, compute_local_rewards):
        step_info = {}
        reward = 0
        local_rewards = None
        for i, metric in enumerate(self.metrics):
            if metric == "eulerian":
                error_fn = self.calculate_eulerian_error
            elif metric == "eulerian_masked":
                error_fn = self.calculate_eulerian_masked_error
            elif metric == "statistical":
                error_fn = self.calculate_statistical_error
            elif metric == "volumetric":
                error_fn = self.calculate_volumetric_error
            elif metric == "height_field":
                error_fn = self.calculate_height_field_error
            elif metric == "potential_energy":
                error_fn = self.calculate_potential_energy_error
            else:
                raise Exception(f"Unrecognized metric: {metric}")
            error, local_errors, baseline, num_samples = error_fn(
                episode_t, compute_local_errors=compute_local_rewards)
            step_info[metric + "_error"] = error
            step_info[metric + "_baseline"] = baseline
            step_info[metric + "_num_samples"] = num_samples
            if self.metrics[metric] > 0:
                reward = -error / num_samples
                if compute_local_rewards:
                    local_rewards = -local_errors
        return reward, local_rewards, step_info

    def step(self, action_converted, compute_local_rewards=False):
        if np.sum(np.isnan(action_converted)) > 0:
            print(action_converted)
            sys.exit(0)

        set_usher_param(self.solver.usher, self.dp, self.unit,
                        self.buoy_v_real, self.coil_x_real, action_converted,
                        self.num_buoys)
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
        self.runner.norm(self.solver.particle_guiding,
                         self.particle_guiding_norm, self.solver.num_particles)
        guiding_mean = self.runner.sum(
            self.particle_guiding_norm,
            self.solver.num_particles) / self.solver.num_particles

        new_state = self.collect_state(self.episode_t + 1)
        # no real unit conversion because arbitrary scaling is employed
        guiding_regularization = guiding_mean / new_state[0, -1] * 2e-6

        # find reward
        reward, local_rewards, step_info = self.calculate_reward(
            self.episode_t + 1, compute_local_rewards)
        reward -= guiding_regularization
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
        return new_state, reward, local_rewards, done, step_info


class EnvironmentPIV(Environment):

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
        super().__init__(dp,
                         truth_dirs,
                         cache_dir,
                         ma_alphas,
                         display,
                         volume_method,
                         save_visual,
                         reward_metric=None,
                         evaluation_metrics=['eulerian_masked'],
                         quick_mode=False)
        self.container_shift = dp.f3(0, self.container_width * 0.5, 0)
        self.pile.x[0] = self.container_shift
        self.cni.grid_offset.y = -4
        self.dp.copy_cn()
        self.buoy_filter_postfix = buoy_filter_postfix
        self.truth_real_freq = 500.0  # TODO: change to 100Hz
        self.truth_real_interval = 1.0 / self.truth_real_freq
        self._max_episode_steps = 2000

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

    def calculate_eulerian_error(self,
                                 episode_t,
                                 use_mask=True,
                                 compute_local_errors=False):
        if use_mask:
            return self.calculate_eulerian_masked_error(
                episode_t, compute_local_errors)
        else:
            raise Exception(
                "Eulerian error without mask is unsupported in PIV")

    def calculate_eulerian_masked_error(self,
                                        episode_t,
                                        compute_local_errors=False):
        local_errors = None
        if compute_local_errors:
            raise Exception("Local errors are unsupported in PIV")
        self.simulation_sampling.prepare_neighbor_and_boundary(
            self.runner, self.solver)
        simulation_v_real = self.simulation_sampling.sample_velocity(
            self.runner, self.solver)
        simulation_v_real.scale(self.unit.to_real_velocity(1))

        self.ground_truth_v.set(self.truth_v_collection[episode_t])
        self.weight.set(self.mask_collection[episode_t])
        error = self.runner.calculate_se_yz_masked(
            simulation_v_real, self.ground_truth_v, self.weight,
            self.simulation_sampling.num_samples)
        baseline = self.runner.calculate_se_yz_masked(
            self.v_zero, self.ground_truth_v, self.weight,
            self.simulation_sampling.num_samples)
        num_samples = np.sum(self.mask_collection[episode_t])

        return error, local_errors, baseline, num_samples
