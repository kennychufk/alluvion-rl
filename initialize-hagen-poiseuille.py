import alluvion as al
import numpy as np
from numpy import linalg as LA
from pathlib import Path
import scipy.special as sc
from sklearn.metrics import mean_squared_error
import glob
import argparse
from optim import AdamOptim

from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Hagen poiseuille initializer')

parser.add_argument('--mode', type=str, default='fill')
args = parser.parse_args()


def approximate_half_life(dynamic_viscosity, density0, pipe_radius):
    j0_zero = sc.jn_zeros(0, 1)[0]
    return np.log(2) / (dynamic_viscosity * j0_zero * j0_zero / density0 /
                        pipe_radius / pipe_radius)


def developing_hagen_poiseuille(r, t, dynamic_viscosity, density0, a,
                                pipe_radius, num_iterations):
    j0_zeros = sc.jn_zeros(0, num_iterations)
    accumulation = np.zeros_like(r)
    for m in range(num_iterations):
        j0_zero = j0_zeros[m]
        j0_zero_ratio = j0_zero / pipe_radius
        j0_zero_ratio_sqr = j0_zero_ratio * j0_zero_ratio

        constant_part = 1 / (j0_zero * j0_zero * j0_zero * sc.jv(1, j0_zero))
        r_dependent_part = sc.jv(0, j0_zero_ratio * r)
        time_dependent_part = np.exp(-dynamic_viscosity * t *
                                     j0_zero_ratio_sqr / density0)
        accumulation += constant_part * r_dependent_part * time_dependent_part
    return density0 * a / dynamic_viscosity * (
        0.25 * (pipe_radius * pipe_radius - r * r) -
        pipe_radius * pipe_radius * 2 * accumulation)


kM = 4
kQ = 10
particle_radius = 2**-11
target_num_particles_range = (23888, 25000)
lambda_factors = np.array([1.125])
accelerations = np.array([1e-3, 5e-4])
dynamic_viscosity = 20e-3

fill_directory = f"Q{kQ}-M{kM}-{particle_radius*1000}mm"
distributed_directory = f"dQ{kQ}-M{kM}-{particle_radius*1000}mm"
Path(fill_directory).mkdir(parents=True, exist_ok=True)
Path(distributed_directory).mkdir(parents=True, exist_ok=True)

dp = al.Depot(np.float32)
cn = dp.cn
cni = dp.cni
dp.create_display(800, 600, "", False)
display_proxy = dp.get_display_proxy()
runner = dp.Runner()

kernel_radius = particle_radius * 4
density0 = 1000.0
cubical_particle_volume = 8 * particle_radius * particle_radius * particle_radius
volume_relative_to_cube = 0.8
particle_mass = cubical_particle_volume * volume_relative_to_cube * density0

cn.set_cubic_discretization_constants()
cn.set_kernel_radius(kernel_radius)
cn.set_particle_attr(particle_radius, particle_mass, density0)
cn.boundary_epsilon = 1e-9
cn.viscosity = 1e-6
cn.boundary_viscosity = cn.viscosity * 4

cylinder_length = 2 * kM * kernel_radius
pipe_radius = kernel_radius * kQ

# rigids
max_num_contacts = 512
pile = dp.Pile(dp, max_num_contacts)
cylinder_distance = dp.InfiniteCylinderDistance(pipe_radius)
pile.add(cylinder_distance,
         al.uint3(64, 1, 64),
         sign=-1,
         thickness=0,
         collision_mesh=al.Mesh(),
         mass=0,
         restitution=1,
         friction=0,
         inertia_tensor=dp.f3(1, 1, 1),
         x=dp.f3(0, 0, 0),
         q=dp.f4(0, 0, 0, 1),
         display_mesh=al.Mesh())

pile.build_grids(4 * kernel_radius)
pile.reallocate_kinematics_on_device()
cn.contact_tolerance = particle_radius

max_num_particles = int(np.pi * pipe_radius * pipe_radius * cylinder_length *
                        density0 / particle_mass)
# grid
grid_res = al.uint3(int(kQ * 2), int(kM * 2), int(kQ * 2))
grid_offset = al.int3(-kQ, -kM, -kQ)
cni.grid_res = grid_res
cni.grid_offset = grid_offset
cni.max_num_particles_per_cell = 64
cni.max_num_neighbors_per_particle = 64
cn.set_wrap_length(grid_res.y * kernel_radius)

solver = dp.SolverDf(runner,
                     pile,
                     dp,
                     max_num_particles,
                     grid_res,
                     enable_surface_tension=False,
                     enable_vorticity=False,
                     graphical=True)
particle_normalized_attr = dp.create_graphical((max_num_particles), 1)
solver.dt = 1e-3
solver.max_dt = 1e-3
solver.min_dt = 0.0
solver.cfl = 2e-2
solver.particle_radius = particle_radius

dp.copy_cn()


class FluidStat:
    def __init__(self):
        self.reset()

    def reset(self):
        self.max_density_error = np.finfo(np.float64).max
        self.min_density_error = np.finfo(np.float64).max
        self.naive_filled_percentage = 0
        self.max_particle_speed = 99.9
        self.min_particle_speed = 99.9
        self.last_stationary_t = 0
        self.step_id = 0
        self.finished = False


class FillState:
    def __init__(self):
        self.next_emission_t = 5.0
        self.last_emission_t = 0
        self.speed_ready_before_emission = False

        self.last_saved_num_particles = 0


class FillSampling:
    def __init__(self, dp, cn, cylinder_length, pipe_radius):
        self.num_sample_slices = 32
        self.num_samples_per_slice = 16
        self.num_samples = self.num_sample_slices * self.num_samples_per_slice
        self.sample_x = dp.create_coated((self.num_samples), 3)
        self.sample_data1 = dp.create_coated((self.num_samples), 1)
        self.sample_neighbors = dp.create_coated(
            (self.num_samples, dp.cni.max_num_neighbors_per_particle), 4)
        self.sample_num_neighbors = dp.create_coated((self.num_samples), 1,
                                                     np.uint32)
        runner.launch_create_fluid_cylinder_sunflower(
            256, self.sample_x, self.num_samples,
            pipe_radius - cn.particle_radius * 2, self.num_samples_per_slice,
            cylinder_length / self.num_sample_slices, cylinder_length * -0.5)


class OptimSampling:
    def __init__(self, dp, cn, cylinder_length, pipe_radius, ts):
        self.ts = ts
        self.num_sample_planes = 14
        self.num_samples_per_plane = 31
        self.num_samples = self.num_samples_per_plane * self.num_sample_planes
        self.sample_x = dp.create_coated((self.num_samples), 3)
        self.sample_data3 = dp.create_coated((self.num_samples), 3)
        self.sample_neighbors = dp.create_coated(
            (self.num_samples, dp.cni.max_num_neighbors_per_particle), 4)
        self.sample_num_neighbors = dp.create_coated((self.num_samples), 1,
                                                     np.uint32)
        self.sample_x_host = np.zeros((self.num_samples, 3), dp.default_dtype)
        distance_between_sample_planes = cylinder_length / self.num_sample_planes
        for i in range(self.num_samples):
            plane_id = i // self.num_samples_per_plane
            id_in_plane = i % self.num_samples_per_plane
            self.sample_x_host[i] = np.array([
                pipe_radius * 2 / (self.num_samples_per_plane + 1) *
                (id_in_plane - self.num_samples_per_plane / 2),
                cylinder_length * -0.5 +
                distance_between_sample_planes * plane_id, 0
            ], dp.default_dtype)
        self.sample_x.set(self.sample_x_host)
        self.reset()

    def reset(self):
        self.sampling_cursor = 0
        self.vx = np.zeros((len(self.ts), self.num_samples_per_plane),
                           dp.default_dtype)


fsampling = FillSampling(dp, cn, cylinder_length, pipe_radius)
fluid_stat = FluidStat()
fill_state = FillState()

colormap_tex = display_proxy.create_colormap_viridis()
display_proxy.add_particle_shading_program(solver.particle_x,
                                           particle_normalized_attr,
                                           colormap_tex,
                                           solver.particle_radius, solver)
display_proxy.set_camera(al.float3(0, 0, pipe_radius * 6), al.float3(0, 0, 0))
display_proxy.set_clip_planes(particle_radius * 10, pipe_radius * 20)


def initialize_uniform():
    naive_num_particles = int(np.pi * (pipe_radius - particle_radius) *
                              (pipe_radius - particle_radius) *
                              cylinder_length * cn.density0 / particle_mass)
    initial_num_particles = naive_num_particles
    slice_distance = particle_radius * 2
    num_slices = int(cylinder_length / slice_distance)
    num_particles_per_slice = initial_num_particles // num_slices
    initial_num_particles = num_particles_per_slice * num_slices

    dp.map_graphical_pointers()
    runner.launch_create_fluid_cylinder_sunflower(
        256, solver.particle_x, initial_num_particles,
        pipe_radius - particle_radius * 2, num_particles_per_slice,
        particle_radius * 2, cylinder_length * -0.5)
    dp.unmap_graphical_pointers()
    solver.num_particles = initial_num_particles


def initialize_with_file(solver, filename):
    dp.map_graphical_pointers()
    solver.num_particles = solver.particle_x.read_file(filename)
    dp.unmap_graphical_pointers()


def reset_solver_df(solver):
    solver.particle_dfsph_factor.set_zero()
    solver.particle_kappa.set_zero()
    solver.particle_kappa_v.set_zero()
    solver.particle_density_adv.set_zero()


def randomize_speed(dp, solver):
    new_particle_v = np.random.normal(0, 0.2, size=(solver.num_particles, 3))
    dp.coat(solver.particle_v).set(new_particle_v)
    reset_solver_df(solver)


def force_stationary(solver, fluid_stat):
    if (fluid_stat.min_particle_speed > 2
            or (fluid_stat.step_id % 10000 == 0)):
        solver.particle_v.set_zero()
        reset_solver_df(solver)
        fluid_stat.last_stationary_t = solver.t
        print(
            f"last stationary t = {fluid_stat.last_stationary_t}, step = {fluid_stat.step_id}, min_particle_speed = {fluid_stat.min_particle_speed}"
        )


def calculate_stat(solver, fluid_stat):
    fluid_stat.max_density_error = runner.max(
        solver.particle_density, solver.num_particles) / cn.density0 - 1
    fluid_stat.min_density_error = runner.min(
        solver.particle_density, solver.num_particles) / cn.density0 - 1
    fluid_stat.max_particle_speed = np.sqrt(
        runner.max(solver.particle_cfl_v2, solver.num_particles))
    fluid_stat.min_particle_speed = np.sqrt(
        runner.min(solver.particle_cfl_v2, solver.num_particles))

    expected_total_volume = np.pi * (pipe_radius - particle_radius) * (
        pipe_radius - particle_radius) * cylinder_length
    fluid_stat.naive_filled_percentage = solver.num_particles * cn.particle_mass / cn.density0 / expected_total_volume


def fill(dp, solver, fluid_stat, fill_state, target_num_particles_range):
    force_stationary(solver, fluid_stat)
    if (not fill_state.speed_ready_before_emission):
        if (fluid_stat.min_density_error < -0.5):
            if (fluid_stat.max_particle_speed < 1e-1):
                fill_state.speed_ready_before_emission = True
        elif (fluid_stat.min_density_error < -0.05):
            if (fluid_stat.max_particle_speed < 5e-3):
                fill_state.speed_ready_before_emission = True
        else:
            if (fluid_stat.max_particle_speed < 1e-2
                    and fluid_stat.min_density_error > -1e-2
                    and fluid_stat.max_density_error < 1e-2):
                fill_state.speed_ready_before_emission = True
    if (solver.num_particles >= target_num_particles_range[0]
            and solver.num_particles <= target_num_particles_range[1]
            and solver.num_particles != fill_state.last_saved_num_particles):
        fill_state.speed_ready_before_emission = False
        if (fluid_stat.max_particle_speed < 2e-3
                and (solver.t - fluid_stat.last_stationary_t > 4)):
            filename = f"{fill_directory}/{solver.num_particles}.alu"
            solver.particle_x.write_file(filename, solver.num_particles)
            fill_state.last_saved_num_particles = solver.num_particles
    if (solver.t > fill_state.next_emission_t
            and fill_state.speed_ready_before_emission
            and solver.num_particles < max_num_particles):
        new_particle_pos = np.array([0, cylinder_length / 2 - 1e-4, 0],
                                    dp.default_dtype)
        new_particle_v = np.array([
            np.random.normal(0, 0.05),
            np.random.normal(0, 0.05),
            np.random.normal(0, 0.05)
        ], dp.default_dtype)
        fill_state.next_emission_t = solver.t + particle_radius * 2 / LA.norm(
            new_particle_v)
        dp.coat(solver.particle_x).set(new_particle_pos, solver.num_particles)
        # TODO: refactor Python binding to change byte_offset to element_offset
        dp.coat(solver.particle_v).set(new_particle_v, solver.num_particles)
        solver.num_particles += 1
        print("num_particles", solver.num_particles)
        fill_state.last_emission_t = solver.t
        fill_state.speed_ready_before_emission = False

    solver.step_wrap1_gravitation1()

    calculate_stat(solver, fluid_stat)

    if (fluid_stat.step_id % 100 == 0):
        solver.pid_length.set_zero()
        runner.launch_update_particle_grid(256, solver.particle_x, solver.pid,
                                           solver.pid_length,
                                           solver.num_particles)
        runner.launch_make_neighbor_list_wrap1(256, fsampling.sample_x,
                                               solver.pid, solver.pid_length,
                                               fsampling.sample_neighbors,
                                               fsampling.sample_num_neighbors,
                                               fsampling.num_samples)
        runner.launch_compute_density(
            256, solver.particle_x, solver.particle_neighbors,
            solver.particle_num_neighbors, solver.particle_density,
            solver.particle_boundary_xj, solver.particle_boundary_volume,
            solver.num_particles)
        runner.launch_sample_fluid(
            256, fsampling.sample_x, solver.particle_x,
            solver.particle_density, solver.particle_density,
            fsampling.sample_neighbors, fsampling.sample_num_neighbors,
            fsampling.sample_data1, fsampling.num_samples)
        print(fsampling.sample_data1.get().reshape(-1, 128)[:, 0])
    fluid_stat.step_id += 1


def distribute(dp, solver, fluid_stat):
    force_stationary(solver, fluid_stat)
    if (fluid_stat.step_id == 0):
        dp.cn.gravity = dp.f3(0, 0, 0)
        dp.copy_cn()
    elif (fluid_stat.step_id == 1000):
        randomize_speed(dp, solver)
    elif (fluid_stat.step_id == 2000):
        cn.gravity = dp.f3(0, 0.02, 0)
        dp.copy_cn()
    elif (fluid_stat.step_id == 4000):
        dp.cn.gravity.y *= -2
        dp.copy_cn()
        print("reversed")
    elif (fluid_stat.step_id == 6000):
        dp.cn.gravity.y *= -2
        dp.copy_cn()
        print("reversed")
    elif (fluid_stat.step_id == 8000):
        dp.cn.gravity = dp.f3(0, 0, 0)
        dp.copy_cn()
        print("no gravity")
    elif (fluid_stat.step_id == 10000):
        solver.particle_v.set_zero()
        reset_solver_df(solver)
        print("stationary")
    elif (fluid_stat.step_id == 11000):
        filename = f"{distributed_directory}/{solver.num_particles}.alu"
        solver.particle_x.write_file(filename, solver.num_particles)
        fluid_stat.finished = True
        print("finished")

    solver.step_wrap1()
    calculate_stat(solver, fluid_stat)
    fluid_stat.step_id += 1


def pressurize(dp, solver, osampling):
    if solver.t >= osampling.ts[osampling.sampling_cursor]:
        solver.pid_length.set_zero()
        runner.launch_update_particle_grid(256, solver.particle_x, solver.pid,
                                           solver.pid_length,
                                           solver.num_particles)
        runner.launch_make_neighbor_list_wrap1(256, osampling.sample_x,
                                               solver.pid, solver.pid_length,
                                               osampling.sample_neighbors,
                                               osampling.sample_num_neighbors,
                                               osampling.num_samples)
        runner.launch_compute_density(
            256, solver.particle_x, solver.particle_neighbors,
            solver.particle_num_neighbors, solver.particle_density,
            solver.particle_boundary_xj, solver.particle_boundary_volume,
            solver.num_particles)
        runner.launch_sample_fluid(256, osampling.sample_x, solver.particle_x,
                                   solver.particle_density, solver.particle_v,
                                   osampling.sample_neighbors,
                                   osampling.sample_num_neighbors,
                                   osampling.sample_data3,
                                   osampling.num_samples)
        sample_vx = osampling.sample_data3.get().reshape(
            -1, osampling.num_samples_per_plane, 3)[:, :, 1]
        osampling.vx[osampling.sampling_cursor] = np.mean(sample_vx, axis=0)
        osampling.sampling_cursor += 1
    solver.step_wrap1()


def evaluate_hagen_poiseuille(dp, solver, x_filename, osampling, viscosity,
                              boundary_viscosity, acc, ts):
    osampling.reset()
    cn.gravity = dp.f3(0, acc, 0)
    cn.viscosity = viscosity
    cn.boundary_viscosity = boundary_viscosity
    dp.copy_cn()
    dp.map_graphical_pointers()
    solver.num_particles = solver.particle_x.read_file(x_filename)
    solver.particle_v.set_zero()
    reset_solver_df(solver)
    solver.dt = 1e-4
    solver.max_dt = 5e-3
    solver.min_dt = 0
    solver.cfl = 0.04
    solver.t = 0
    solver.particle_x
    step_id = 0
    while osampling.sampling_cursor < len(ts):
        pressurize(dp, solver, osampling)
        step_id += 1
        # if (step_id % 20 == 0):
        #     solver.normalize(solver.particle_v, particle_normalized_attr, 0, 0.5)
        #     dp.unmap_graphical_pointers()
        #     display_proxy.draw()
        #     dp.map_graphical_pointers()
    dp.unmap_graphical_pointers()
    return osampling.vx


# across a variety of speed. Fixed radius. Fixed target dynamic viscosity. Look at one instant
def compute_ground_truth_and_simulate(param, dp, solver, osampling, x_filename,
                                      pipe_radius, ts, density0, accelerations,
                                      dynamic_viscosity):
    viscosity = param[0]
    boundary_viscosity = param[1]

    num_samples_per_side = osampling.num_samples_per_plane // 2
    rs = pipe_radius * 2 / (osampling.num_samples_per_plane + 1) * (
        np.arange(osampling.num_samples_per_plane) - num_samples_per_side)

    ground_truth = np.zeros(
        (len(accelerations), len(ts), osampling.num_samples_per_plane))
    simulated = np.zeros_like(ground_truth)
    for acc_id, acc in enumerate(accelerations):
        for t_id, t in enumerate(ts):
            ground_truth[acc_id, t_id] = developing_hagen_poiseuille(
                rs, t, dynamic_viscosity, density0, acc, pipe_radius, 100)

        result = evaluate_hagen_poiseuille(dp, solver, x_filename, osampling,
                                           viscosity, boundary_viscosity, acc,
                                           ts)
        simulated[acc_id] = result.reshape(len(ts), -1)

    # fig = plt.figure(figsize=(4, 5), dpi=110)
    # ax = fig.add_subplot()
    # plt.plot(ground_truth[0][0])
    # plt.show()
    # plt.plot(simulated[0][0])
    # plt.show()
    return ground_truth, simulated


def evaluate_loss(param, dp, solver, osampling, x_filename, pipe_radius, ts,
                  density0, accelerations, dynamic_viscosity):
    ground_truth, simulated = compute_ground_truth_and_simulate(
        param, dp, solver, osampling, x_filename, pipe_radius, ts, density0,
        accelerations, dynamic_viscosity)
    return mean_squared_error(ground_truth.flatten(), simulated.flatten())


def optimize(dp, solver, num_particles, dynamic_viscosity):
    x_filename = f"{distributed_directory}/{num_particles}.alu"
    approx_half_life = approximate_half_life(dynamic_viscosity, density0,
                                             pipe_radius)
    ts = approx_half_life * lambda_factors
    osampling = OptimSampling(dp, cn, cylinder_length, pipe_radius, ts)

    best_loss = np.finfo(np.float64).max
    best_x = None
    x = np.array([1.37916076e-05, 9.96210578e-06])
    adam = AdamOptim(x, lr=1e-9)

    for iteration in range(100):
        current_x = x
        x, loss, grad = adam.update(evaluate_loss, x,
                                    np.min(x) * 1e-2, dp, solver, osampling,
                                    x_filename, pipe_radius, lambda_factors,
                                    density0, accelerations, dynamic_viscosity)
        if (loss < best_loss):
            best_loss = loss
            best_x = current_x
        print(current_x, x, loss, grad)
        print('best x', best_x)


if args.mode == 'fill':
    initialize_uniform()
    cn.axial_gravity = -10
    cn.radial_gravity = 5.0
    dp.copy_cn()
    while not fluid_stat.finished:
        dp.map_graphical_pointers()
        for frame_interstep in range(10):
            fill(dp, solver, fluid_stat, fill_state,
                 target_num_particles_range)
        solver.normalize(solver.particle_v, particle_normalized_attr, 0, 0.5)
        dp.unmap_graphical_pointers()
        display_proxy.draw()
elif args.mode == 'distribute':
    for fill_filename in glob.glob(f"{fill_directory}/*.alu"):
        initialize_with_file(solver, fill_filename)
        fluid_stat.reset()
        dp.copy_cn()
        while not fluid_stat.finished:
            dp.map_graphical_pointers()
            for frame_interstep in range(10):
                distribute(dp, solver, fluid_stat)
            solver.normalize(solver.particle_v, particle_normalized_attr, 0,
                             0.5)
            dp.unmap_graphical_pointers()
            display_proxy.draw()
elif args.mode == 'optimize':
    optimize(dp, solver, 24114, dynamic_viscosity)
