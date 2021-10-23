import glob
import argparse
import subprocess
import os

import alluvion as al
import numpy as np
from numpy import linalg as LA
from pathlib import Path
import scipy.special as sc
from sklearn.metrics import mean_squared_error, mean_absolute_error
from optim import AdamOptim
from util import Unit, FluidSample

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import patheffects as path_effects

import wandb

parser = argparse.ArgumentParser(description='Hagen poiseuille initializer')

parser.add_argument('pos', metavar='x', type=str, nargs=1)
parser.add_argument('--display', metavar='d', type=bool, default=False)
args = parser.parse_args()


def approximate_half_life(kinematic_viscosity, pipe_radius):
    j0_zero = sc.jn_zeros(0, 1)[0]
    return np.log(2) / (kinematic_viscosity * j0_zero * j0_zero / pipe_radius /
                        pipe_radius)


def developing_hagen_poiseuille(r, t, kinematic_viscosity, a, pipe_radius,
                                num_iterations):
    j0_zeros = sc.jn_zeros(0, num_iterations)
    accumulation = np.zeros_like(r)
    for m in range(num_iterations):
        j0_zero = j0_zeros[m]
        j0_zero_ratio = j0_zero / pipe_radius
        j0_zero_ratio_sqr = j0_zero_ratio * j0_zero_ratio

        constant_part = 1 / (j0_zero * j0_zero * j0_zero * sc.jv(1, j0_zero))
        r_dependent_part = sc.jv(0, j0_zero_ratio * r)
        time_dependent_part = np.exp(-kinematic_viscosity * t *
                                     j0_zero_ratio_sqr)
        accumulation += constant_part * r_dependent_part * time_dependent_part
    return a / kinematic_viscosity * (
        0.25 * (pipe_radius * pipe_radius - r * r) -
        pipe_radius * pipe_radius * 2 * accumulation)


def acceleration_from_terminal_velocity(terminal_v, kinematic_viscosity,
                                        pipe_radius):
    return terminal_v * 4 * kinematic_viscosity / (pipe_radius * pipe_radius)


def calculate_terminal_velocity(kinematic_viscosity, pipe_radius,
                                accelerations):
    return 0.25 * accelerations * pipe_radius * pipe_radius / kinematic_viscosity


class OptimSampling(FluidSample):
    def __init__(self, dp, pipe_length, pipe_radius, ts, num_particles):
        self.ts = ts
        self.num_sections = 14
        self.num_rotations = 16
        self.num_rs = 16  # should be even number
        self.num_particles = num_particles

        # r contains 0 but not pipe_radius
        self.rs = pipe_radius / self.num_rs * np.arange(self.num_rs)

        num_samples = self.num_rs * self.num_sections * self.num_rotations
        sample_x_host = np.zeros((num_samples, 3), dp.default_dtype)
        section_length = pipe_length / self.num_sections
        offset_y_per_rotation = section_length / self.num_rotations
        theta_per_rotation = np.pi * 2 / self.num_rotations
        for i in range(num_samples):
            section_id = i // (self.num_rs * self.num_rotations)
            rotation_id = (i // self.num_rs) % (self.num_rotations)
            r_id = i % self.num_rs
            theta = theta_per_rotation * rotation_id
            sample_x_host[i] = np.array([
                self.rs[r_id] * np.cos(theta),
                pipe_length * -0.5 + section_length * section_id +
                offset_y_per_rotation * rotation_id,
                self.rs[r_id] * np.sin(theta)
            ], dp.default_dtype)
        super().__init__(dp, sample_x_host)
        self.reset()

    def reset(self):
        self.sampling_cursor = 0
        self.vx = np.zeros((len(self.ts), self.num_rs), self.dp.default_dtype)
        self.density = np.zeros((len(self.ts), self.num_rs),
                                self.dp.default_dtype)
        self.r_stat = np.zeros((len(self.ts), self.num_particles),
                               self.dp.default_dtype)
        self.density_stat = np.zeros((len(self.ts), self.num_particles),
                                     self.dp.default_dtype)

    def aggregate(self):
        sample_vx = self.sample_data3.get().reshape(-1, self.num_rs, 3)[..., 1]
        self.vx[self.sampling_cursor] = np.mean(sample_vx, axis=0)

        density_ungrouped = self.sample_data1.get().reshape(-1, self.num_rs)
        self.density[self.sampling_cursor] = np.mean(density_ungrouped, axis=0)
        self.sampling_cursor += 1


class TemporalStat:
    def __init__(self):
        self.ts = []
        self.temporal_dict = {}
        self.radial_density = None
        self.rs = None
        self.density_stat = None
        self.r_stat = None

    def append(self, t, **kwargs):
        self.ts.append(t)
        for key in kwargs:
            if key not in self.temporal_dict:
                self.temporal_dict[key] = []
            self.temporal_dict[key].append(kwargs[key])


def pressurize(dp, solver, osampling):
    next_sample_t = osampling.ts[osampling.sampling_cursor]
    if solver.t + solver.max_dt >= next_sample_t:
        remainder_dt = next_sample_t - solver.t
        original_dt = solver.max_dt
        solver.max_dt = solver.min_dt = solver.initial_dt = remainder_dt
        solver.step_wrap1()
        solver.max_dt = solver.min_dt = solver.initial_dt = original_dt
        osampling.prepare_neighbor_and_boundary_wrap1(runner, solver)
        osampling.sample_vector3(runner, solver, solver.particle_v)
        osampling.sample_density(runner)
        osampling.density_stat[osampling.sampling_cursor] = dp.coat(
            solver.particle_density).get()
        osampling.r_stat[osampling.sampling_cursor] = LA.norm(dp.coat(
            solver.particle_x).get()[:, [0, 2]],
                                                              axis=1)
        osampling.aggregate()
    else:
        solver.step_wrap1()


def evaluate_hagen_poiseuille(dp, solver, initial_particle_x, osampling, param,
                              is_grad_eval, acc, ts):
    osampling.reset()
    tstat = TemporalStat()
    cn.gravity = dp.f3(0, acc, 0)
    cn.viscosity = param[0]
    cn.boundary_viscosity = param[1]
    # solver.enable_divergence_solve = False
    # solver.enable_density_solve = False
    cn.dfsph_factor_epsilon = 1e-6
    solver.max_density_solve = 0
    solver.min_density_solve = 0
    # cn.vorticity_coeff = param[2]
    # cn.viscosity_omega = param[3]
    cn.inertia_inverse = 0.5  # recommended by author
    dp.copy_cn()
    dp.map_graphical_pointers()
    solver.particle_x.set_from(initial_particle_x)
    solver.particle_v.set_zero()
    solver.reset_solving_var()
    solver.reset_t()
    solver.num_particles = initial_particle_x.get_shape()[0]
    step_id = 0
    while osampling.sampling_cursor < len(ts):
        pressurize(dp, solver, osampling)
        if not is_grad_eval:
            density_min = dp.Runner.min(solver.particle_density,
                                        solver.num_particles)
            density_max = dp.Runner.max(solver.particle_density,
                                        solver.num_particles)
            density_mean = dp.Runner.sum(
                solver.particle_density,
                solver.num_particles) / solver.num_particles
            tstat.append(solver.t,
                         cfl_dt=solver.cfl_dt,
                         dt=solver.dt,
                         density_min=density_min,
                         density_max=density_max,
                         density_mean=density_mean)
        # if step_id % 1000 == 0:
        #     print('cfl dt:', solver.cfl_dt, 'dt:', solver.dt, 'v2:', solver.max_v2)
        if (dp.has_display() and step_id % 20 == 0):
            dp.get_display_proxy().draw()
        step_id += 1
    dp.unmap_graphical_pointers()
    tstat.radial_density = osampling.density
    tstat.rs = osampling.rs
    tstat.density_stat = osampling.density_stat
    tstat.r_stat = osampling.r_stat
    return osampling.vx, tstat


# across a variety of speed. Fixed radius. Fixed target dynamic viscosity. Look at one instant
def compute_ground_truth(osampling, pipe_radius, ts, accelerations,
                         kinematic_viscosity):
    ground_truth = np.zeros((len(accelerations), len(ts), osampling.num_rs))
    for acc_id, acc in enumerate(accelerations):
        for t_id, t in enumerate(ts):
            ground_truth[acc_id, t_id] = developing_hagen_poiseuille(
                osampling.rs, t, kinematic_viscosity, acc, pipe_radius, 100)
    return ground_truth


def simulate(param, is_grad_eval, dp, solver, osampling, initial_particle_x,
             pipe_radius, ts, accelerations, kinematic_viscosity):
    simulated = np.zeros((len(accelerations), len(ts), osampling.num_rs))
    summary = []
    for acc_id, acc in enumerate(accelerations):
        result, tstat = evaluate_hagen_poiseuille(dp, solver,
                                                  initial_particle_x,
                                                  osampling, param,
                                                  is_grad_eval, acc, ts)
        simulated[acc_id] = result
        summary.append(tstat)
    return simulated, summary


def mse_loss(ground_truth, simulated):
    # scale = np.repeat(np.max(ground_truth, axis=2)[..., np.newaxis], ground_truth.shape[-1], axis=2)
    # ground_truth_scaled = ground_truth / scale
    # simulated_scaled = simulated / scale
    return mean_squared_error(ground_truth.flatten(), simulated.flatten())


def save_result(iteration, ground_truth, simulated, osampling, unit,
                accelerations):
    my_dpi = 128
    num_rows = 1
    num_cols = len(accelerations)
    fig = plt.figure(figsize=(1024 / my_dpi, 768 / my_dpi), dpi=my_dpi)
    cmap = plt.get_cmap("tab10")
    for acc_id, acc in enumerate(accelerations):
        ax = fig.add_subplot(num_rows, num_cols, acc_id + 1)
        ax.set_title(f'a = {unit.to_real_acceleration(acc):.3e}ms-2')
        ax.set_ylabel('v (m/s)')
        for t_id, t in enumerate(osampling.ts):
            ax.plot(unit.to_real_length(osampling.rs),
                    unit.to_real_velocity(ground_truth[acc_id][t_id]),
                    c=cmap(t_id),
                    linewidth=5,
                    alpha=0.3)
            ax.plot(unit.to_real_length(osampling.rs),
                    unit.to_real_velocity(simulated[acc_id][t_id]),
                    c=cmap(t_id),
                    label=f"{unit.to_real_time(t):.2f}s")
        ax.set_xlabel('r (m)')
        ax.legend()
    plt.savefig(f'.alcache/{iteration}.png')
    plt.close('all')


def plot_summary(iteration, summary, unit):
    my_dpi = 128
    num_rows = 1
    num_cols = len(accelerations)
    fig = plt.figure(figsize=(1024 / my_dpi, 768 / my_dpi), dpi=my_dpi)
    for acc_id, acc in enumerate(accelerations):
        ax = fig.add_subplot(num_rows, num_cols, acc_id + 1)
        tstat = summary[acc_id]
        keys = ['density_min', 'density_max', 'density_mean']
        for key in keys:
            ax.plot(unit.to_real_time(np.array(tstat.ts)),
                    unit.to_real_density(np.array(tstat.temporal_dict[key])),
                    label=key)
        ax.set_xlabel('t (s)')
        ax.legend()
    plt.savefig(f'.alcache/density{iteration}.png')
    plt.close('all')

    fig = plt.figure(figsize=(1024 / my_dpi, 768 / my_dpi), dpi=my_dpi)
    for acc_id, acc in enumerate(accelerations):
        ax = fig.add_subplot(num_rows, num_cols, acc_id + 1)
        tstat = summary[acc_id]
        for t_id in range(len(tstat.radial_density)):
            ax.plot(unit.to_real_time(np.array(tstat.rs)),
                    unit.to_real_density(np.array(tstat.radial_density[t_id])),
                    label=f"{t_id}")
        ax.set_xlabel('r (m)')
        ax.legend()
    plt.savefig(f'.alcache/radial_density{iteration}.png')
    plt.close('all')

    fig = plt.figure(figsize=(1024 / my_dpi, 768 / my_dpi), dpi=my_dpi)
    for acc_id, acc in enumerate(accelerations):
        ax = fig.add_subplot(num_rows, num_cols, acc_id + 1)
        tstat = summary[acc_id]
        for t_id in range(len(tstat.density_stat)):
            ax.scatter(unit.to_real_length(np.array(tstat.r_stat[t_id])),
                       unit.to_real_density(np.array(
                           tstat.density_stat[t_id])),
                       label=f"{t_id}",
                       s=0.5)
        ax.set_xlabel('r (m)')
        ax.legend()
    plt.savefig(f'.alcache/density_scatter{iteration}.png')
    plt.close('all')


def make_animate_command(input_filename, output_filename):
    return [
        "ffmpeg", "-i", input_filename, "-r", "30.00", "-c:v", "libx264",
        "-crf", "21", "-pix_fmt", "yuv420p", output_filename
    ]


def optimize(dp, solver, adam, param0, initial_particle_x, unit, pipe_radius,
             lambda_factors, accelerations, kinematic_viscosity):
    approx_half_life = approximate_half_life(kinematic_viscosity, pipe_radius)
    print('approx_half_life', approx_half_life)
    ts = approx_half_life * lambda_factors

    print('ts', ts)
    osampling = OptimSampling(dp, pipe_length, pipe_radius, ts,
                              solver.num_particles)

    best_loss = np.finfo(np.float64).max
    best_x = None
    x = param0
    ground_truth = compute_ground_truth(osampling, pipe_radius, ts,
                                        accelerations, kinematic_viscosity)

    with open('switch', 'w') as f:
        f.write('1')
    for iteration in range(100):
        with open('switch', 'r') as f:
            if f.read(1) == '0':
                break
        current_x = x
        x, loss, grad, simulated, summary = adam.update(
            simulate, ground_truth, mse_loss, x, x * 1e-2, dp, solver,
            osampling, initial_particle_x, pipe_radius, ts, accelerations,
            kinematic_viscosity)
        if (loss < best_loss):
            best_loss = loss
            wandb.summary['best_loss'] = best_loss
            best_x = current_x
            wandb.summary['best_x'] = best_x
        save_result(iteration, ground_truth, simulated, osampling, unit,
                    accelerations)
        plot_summary(iteration, summary, unit)
        param_names = ['vis', 'bvis']
        log_object = {'loss': loss, '|∇|': LA.norm(grad)}
        for param_id, param_value in enumerate(current_x):
            log_object[param_names[param_id]] = param_value
            log_object[param_names[param_id] +
                       '_real'] = unit.to_real_kinematic_viscosity(param_value)
            log_object['∇' + param_names[param_id]] = grad[param_id]
        wandb.log(log_object)
    ## finalize
    subprocess.Popen(make_animate_command(".alcache/%d.png",
                                          ".alcache/profile.mp4"),
                     env=os.environ.copy()).wait()
    subprocess.Popen(make_animate_command(".alcache/density%d.png",
                                          ".alcache/density_stat.mp4"),
                     env=os.environ.copy()).wait()
    subprocess.Popen(make_animate_command(".alcache/radial_density%d.png",
                                          ".alcache/radial_density.mp4"),
                     env=os.environ.copy()).wait()
    subprocess.Popen(make_animate_command(".alcache/density_scatter%d.png",
                                          ".alcache/density_scatter.mp4"),
                     env=os.environ.copy()).wait()
    wandb.log({
        "profile":
        wandb.Video('.alcache/profile.mp4', fps=30, format="mp4"),
        "density_stat":
        wandb.Video('.alcache/density_stat.mp4', fps=30, format="mp4"),
        "radial_density":
        wandb.Video('.alcache/radial_density.mp4', fps=30, format="mp4"),
        "density_scatter":
        wandb.Video('.alcache/density_scatter.mp4', fps=30, format="mp4")
    })


dp = al.Depot(np.float32)
cn = dp.cn
cni = dp.cni
if args.display:
    dp.create_display(800, 600, "", False)
display_proxy = dp.get_display_proxy() if args.display else None
runner = dp.Runner()

# Physical constants
density0 = 1

# Model parameters
particle_radius = 0.25
kernel_radius = particle_radius * 4
cubical_particle_volume = 8 * particle_radius * particle_radius * particle_radius
volume_relative_to_cube = 0.8
particle_mass = cubical_particle_volume * volume_relative_to_cube * density0

cn.set_kernel_radius(kernel_radius)
cn.set_particle_attr(particle_radius, particle_mass, density0)

# Real-world unit conversion
unit = Unit(real_kernel_radius=0.0025,
            real_density0=1000,
            real_gravity=-9.80665)
kinematic_viscosity_real = np.load('kinematic_viscosity_real.npy').item()
kinematic_viscosity = unit.from_real_kinematic_viscosity(
    kinematic_viscosity_real)

# Pipe dimension
pipe_radius_grid_span = 10
initial_radius = kernel_radius * pipe_radius_grid_span
pipe_model_radius = 7.69211562500019  # tight radius (7.69211562500019) # TODO: read from stat
pipe_length_grid_half_span = 3
pipe_length_half = pipe_length_grid_half_span * kernel_radius
pipe_length = 2 * pipe_length_grid_half_span * kernel_radius
num_particles = dp.Runner.get_fluid_cylinder_num_particles(
    initial_radius, -pipe_length_half, pipe_length_half, particle_radius)
pipe_volume = num_particles * cn.particle_vol
pipe_radius = np.sqrt(pipe_volume / pipe_length / np.pi)
print('num_particles', num_particles)
print('pipe_radius', pipe_radius)

# Pressurization and temporal parameters
lambda_factors = np.array([0.5, 1.125, 1.5])
accelerations = np.array(
    [1.0]) * 4 * kinematic_viscosity * kinematic_viscosity / (
        pipe_radius * pipe_radius * pipe_radius)
print('accelerations', accelerations)
# accelerations = np.array([2**-13 / 100])
# accelerations = np.array([2**-8 / 100])
terminal_v = calculate_terminal_velocity(kinematic_viscosity, pipe_radius,
                                         accelerations)
print('terminal_v', terminal_v)

# rigids
max_num_contacts = 512
pile = dp.Pile(dp, runner, max_num_contacts)
pile.add(dp.InfiniteCylinderDistance.create(pipe_model_radius),
         al.uint3(64, 1, 64),
         sign=-1)

pile.reallocate_kinematics_on_device()
cn.contact_tolerance = particle_radius

# grid
cni.grid_res = al.uint3(int(pipe_radius_grid_span * 2),
                        int(pipe_length_grid_half_span * 2),
                        int(pipe_radius_grid_span * 2))
cni.grid_offset = al.int3(-pipe_radius_grid_span, -pipe_length_grid_half_span,
                          -pipe_radius_grid_span)

cni.max_num_particles_per_cell = 64
cni.max_num_neighbors_per_particle = 64
cn.set_wrap_length(cni.grid_res.y * kernel_radius)

solver = dp.SolverI(runner,
                    pile,
                    dp,
                    num_particles,
                    enable_surface_tension=False,
                    enable_vorticity=False,
                    graphical=args.display)
solver.num_particles = num_particles
# solver.max_dt = 0.2
# solver.max_dt = 0.1 * real_kernel_radius * inv_real_time
solver.cfl = 0.015625
solver.max_dt = solver.cfl * particle_radius * 2 / terminal_v
solver.min_dt = solver.max_dt
print('dt', solver.max_dt)
solver.initial_dt = solver.max_dt
# solver.density_change_tolerance = 1e-4
# solver.density_error_tolerance = 1e-4

if args.display:
    display_proxy.set_camera(al.float3(0, 0, pipe_radius * 6),
                             al.float3(0, 0, 0))
    display_proxy.set_clip_planes(particle_radius * 10, pipe_radius * 20)
    colormap_tex = display_proxy.create_colormap_viridis()
    particle_normalized_attr = dp.create_graphical((num_particles), 1)

    display_proxy.add_normalize(solver, solver.particle_v,
                                particle_normalized_attr, 0, 0.5)
    display_proxy.add_unmap_graphical_pointers(dp)
    display_proxy.add_particle_shading_program(solver.particle_x,
                                               particle_normalized_attr,
                                               colormap_tex,
                                               solver.particle_radius, solver)
    display_proxy.add_map_graphical_pointers(dp)

initial_particle_x = dp.create((num_particles), 3)
initial_particle_x.read_file(args.pos[0])
param0 = unit.from_real_kinematic_viscosity(np.array(
    [2.049, 6.532])) * kinematic_viscosity_real
adam = AdamOptim(param0, lr=2e-4)

old_filenames = glob.glob('.alcache/*.mp4') + glob.glob('.alcache/*.png')
for filename in old_filenames:
    try:
        os.remove(filename)
    except:
        print("Error while deleting file : ", filename)

wandb.init(project='alluvion', tags=['vis_sweep2'])
config = wandb.config
config.kinematic_viscosity_real = kinematic_viscosity_real
config.pipe_radius_real = unit.to_real_length(pipe_radius)
config.accelerations_real = unit.to_real_acceleration(accelerations)
config.num_particles = num_particles
config.half_life_real = unit.to_real_time(
    approximate_half_life(kinematic_viscosity, pipe_radius))
config.lambda_factors = lambda_factors
config.kernel_radius_real = unit.to_real_length(kernel_radius)
config.pipe_mode_radius_real = unit.to_real_length(pipe_model_radius)
config.precision = str(dp.default_dtype)
config.Re = pipe_radius * pipe_radius * pipe_radius * 0.25 / kinematic_viscosity / kinematic_viscosity * accelerations
print('Re', config.Re)
config.lr = adam.lr
config.initial_dt = solver.initial_dt
config.max_dt = solver.max_dt
config.min_dt = solver.min_dt
config.cfl = solver.cfl
# config.density_change_tolerance = solver.density_change_tolerance
config.density_error_tolerance = solver.density_error_tolerance

optimize(dp, solver, adam, param0, initial_particle_x, unit, pipe_radius,
         lambda_factors, accelerations, kinematic_viscosity)
