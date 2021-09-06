import glob
import argparse
import subprocess
import os

import alluvion as al
import numpy as np
from numpy import linalg as LA
from pathlib import Path
import scipy.special as sc
from sklearn.metrics import mean_squared_error
from optim import AdamOptim

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import patheffects as path_effects

import wandb

parser = argparse.ArgumentParser(description='Hagen poiseuille initializer')

parser.add_argument('pos', metavar='x', type=str, nargs=1)
parser.add_argument('--display', metavar='d', type=bool, default=False)
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


class OptimSampling:
    def __init__(self, dp, cn, pipe_length, pipe_radius, ts, num_particles):
        self.ts = ts
        self.num_sections = 14
        self.num_rotations = 16
        self.num_rs = 16  # should be even number
        self.num_samples = self.num_rs * self.num_sections * self.num_rotations
        self.num_particles = num_particles

        # r contains 0 but not pipe_radius
        self.rs = pipe_radius / self.num_rs * np.arange(self.num_rs)

        self.sample_x = dp.create_coated((self.num_samples), 3)
        self.sample_data3 = dp.create_coated((self.num_samples), 3)
        self.sample_density = dp.create_coated((self.num_samples), 1)
        self.sample_boundary = dp.create_coated(
            (dp.cni.num_boundaries, self.num_samples), 4)
        self.sample_boundary_kernel = dp.create_coated(
            (dp.cni.num_boundaries, self.num_samples), 4)
        self.sample_neighbors = dp.create_coated(
            (self.num_samples, dp.cni.max_num_neighbors_per_particle), 4)
        self.sample_num_neighbors = dp.create_coated((self.num_samples), 1,
                                                     np.uint32)
        sample_x_host = np.zeros((self.num_samples, 3), dp.default_dtype)
        section_length = pipe_length / self.num_sections
        offset_y_per_rotation = section_length / self.num_rotations
        theta_per_rotation = np.pi * 2 / self.num_rotations
        for i in range(self.num_samples):
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
        self.sample_x.set(sample_x_host)
        self.dp = dp
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

        density_ungrouped = self.sample_density.get().reshape(-1, self.num_rs)
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
    if solver.t >= osampling.ts[osampling.sampling_cursor]:
        solver.update_particle_neighbors_wrap1()
        runner.launch_make_neighbor_list_wrap1(osampling.sample_x, solver.pid,
                                               solver.pid_length,
                                               osampling.sample_neighbors,
                                               osampling.sample_num_neighbors,
                                               osampling.num_samples)
        solver.sample_all_boundaries(osampling.sample_x,
                                     osampling.sample_boundary,
                                     osampling.sample_boundary_kernel,
                                     osampling.num_samples)
        runner.launch_sample_fluid(osampling.sample_x, solver.particle_x,
                                   solver.particle_density, solver.particle_v,
                                   osampling.sample_neighbors,
                                   osampling.sample_num_neighbors,
                                   osampling.sample_data3,
                                   osampling.num_samples)
        runner.launch_sample_density(osampling.sample_x,
                                     osampling.sample_neighbors,
                                     osampling.sample_num_neighbors,
                                     osampling.sample_density,
                                     osampling.sample_boundary_kernel,
                                     osampling.num_samples)
        osampling.density_stat[osampling.sampling_cursor] = dp.coat(
            solver.particle_density).get()
        osampling.r_stat[osampling.sampling_cursor] = LA.norm(dp.coat(
            solver.particle_x).get()[:, [0, 2]],
                                                              axis=1)
        osampling.aggregate()
    solver.step_wrap1()


def evaluate_hagen_poiseuille(dp, solver, initial_particle_x, osampling, param,
                              is_grad_eval, acc, ts):
    osampling.reset()
    tstat = TemporalStat()
    cn.gravity = dp.f3(0, acc, 0)
    cn.viscosity = param[0]
    cn.boundary_viscosity = param[1]
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
        step_id += 1
        if (dp.has_display() and step_id % 20 == 0):
            dp.get_display_proxy().draw()
    dp.unmap_graphical_pointers()
    tstat.radial_density = osampling.density
    tstat.rs = osampling.rs
    tstat.density_stat = osampling.density_stat
    tstat.r_stat = osampling.r_stat
    return osampling.vx, tstat


# across a variety of speed. Fixed radius. Fixed target dynamic viscosity. Look at one instant
def compute_ground_truth(osampling, pipe_radius, ts, density0, accelerations,
                         dynamic_viscosity):
    ground_truth = np.zeros((len(accelerations), len(ts), osampling.num_rs))
    for acc_id, acc in enumerate(accelerations):
        for t_id, t in enumerate(ts):
            ground_truth[acc_id, t_id] = developing_hagen_poiseuille(
                osampling.rs, t, dynamic_viscosity, density0, acc, pipe_radius,
                100)
    return ground_truth


def simulate(param, is_grad_eval, dp, solver, osampling, initial_particle_x,
             pipe_radius, ts, density0, accelerations, dynamic_viscosity):
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
    return mean_squared_error(ground_truth.flatten(), simulated.flatten())


def save_result(iteration, ground_truth, simulated, osampling, accelerations):
    my_dpi = 128
    num_rows = 1
    num_cols = len(accelerations)
    fig = plt.figure(figsize=(1024 / my_dpi, 768 / my_dpi), dpi=my_dpi)
    cmap = plt.get_cmap("tab10")
    for acc_id, acc in enumerate(accelerations):
        ax = fig.add_subplot(num_rows, num_cols, acc_id + 1)
        for t_id, t in enumerate(osampling.ts):
            ax.plot(osampling.rs,
                    ground_truth[acc_id][t_id],
                    c=cmap(t_id),
                    linewidth=5,
                    alpha=0.3)
            ax.plot(osampling.rs,
                    simulated[acc_id][t_id],
                    c=cmap(t_id),
                    label=f"{t:.2f}s")
        ax.set_xlabel('r (m)')
        ax.legend()
    plt.savefig(f'.alcache/{iteration}.png')
    plt.close('all')


def plot_summary(iteration, summary):
    my_dpi = 128
    num_rows = 1
    num_cols = len(accelerations)
    fig = plt.figure(figsize=(1024 / my_dpi, 768 / my_dpi), dpi=my_dpi)
    for acc_id, acc in enumerate(accelerations):
        ax = fig.add_subplot(num_rows, num_cols, acc_id + 1)
        tstat = summary[acc_id]
        keys = ['density_min', 'density_max', 'density_mean']
        for key in keys:
            ax.plot(tstat.ts, tstat.temporal_dict[key], label=key)
        ax.set_xlabel('t (s)')
        ax.legend()
    plt.savefig(f'.alcache/density{iteration}.png')
    plt.close('all')

    fig = plt.figure(figsize=(1024 / my_dpi, 768 / my_dpi), dpi=my_dpi)
    for acc_id, acc in enumerate(accelerations):
        ax = fig.add_subplot(num_rows, num_cols, acc_id + 1)
        tstat = summary[acc_id]
        for t_id in range(len(tstat.radial_density)):
            ax.plot(tstat.rs, tstat.radial_density[t_id], label=f"{t_id}")
        ax.set_xlabel('r (m)')
        ax.legend()
    plt.savefig(f'.alcache/radial_density{iteration}.png')
    plt.close('all')

    fig = plt.figure(figsize=(1024 / my_dpi, 768 / my_dpi), dpi=my_dpi)
    for acc_id, acc in enumerate(accelerations):
        ax = fig.add_subplot(num_rows, num_cols, acc_id + 1)
        tstat = summary[acc_id]
        for t_id in range(len(tstat.density_stat)):
            ax.scatter(tstat.r_stat[t_id],
                       tstat.density_stat[t_id],
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


def optimize(dp, solver, adam, param0, initial_particle_x, pipe_radius,
             lambda_factors, density0, accelerations, dynamic_viscosity):
    approx_half_life = approximate_half_life(dynamic_viscosity, density0,
                                             pipe_radius)
    print('approx_half_life', approx_half_life)
    ts = approx_half_life * lambda_factors
    osampling = OptimSampling(dp, cn, pipe_length, pipe_radius, ts,
                              solver.num_particles)

    best_loss = np.finfo(np.float64).max
    best_x = None
    x = param0
    ground_truth = compute_ground_truth(osampling, pipe_radius, ts, density0,
                                        accelerations, dynamic_viscosity)

    with open('switch', 'w') as f:
        f.write('1')
    for iteration in range(200):
        with open('switch', 'r') as f:
            if f.read(1) == '0':
                break
        current_x = x
        x, loss, grad, simulated, summary = adam.update(
            simulate, ground_truth, mse_loss, x, x * 1e-2, dp, solver,
            osampling, initial_particle_x, pipe_radius, ts, density0,
            accelerations, dynamic_viscosity)
        if (loss < best_loss):
            best_loss = loss
            wandb.summary['best_loss'] = best_loss
            best_x = current_x
            wandb.summary['best_x'] = best_x
        save_result(iteration, ground_truth, simulated, osampling,
                    accelerations)
        plot_summary(iteration, summary)
        param_names = ['vis', 'bvis']
        log_object = {'loss': loss, '|∇|': LA.norm(grad)}
        for param_id, param_value in enumerate(current_x):
            log_object[param_names[param_id]] = param_value
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
dynamic_viscosity = 0.01

# Model parameters
scale_factor = 1
particle_radius = 0.25 * scale_factor
kernel_radius = particle_radius * 4
cubical_particle_volume = 8 * particle_radius * particle_radius * particle_radius
volume_relative_to_cube = 0.8
particle_mass = cubical_particle_volume * volume_relative_to_cube * density0

cn.set_cubic_discretization_constants()
cn.set_kernel_radius(kernel_radius)
cn.set_particle_attr(particle_radius, particle_mass, density0)

# Pipe dimension
pipe_radius_grid_span = 10
initial_radius = kernel_radius * pipe_radius_grid_span
pipe_model_radius = 7.69324 * scale_factor  # 7.69324 (more stable at Re=1) for 9900 particles. Larger than the experimental value (7.69211562500019)
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
accelerations = np.array([2**-13 / 100])

# rigids
max_num_contacts = 512
pile = dp.Pile(dp, runner, max_num_contacts)
pile.add(dp.InfiniteCylinderDistance.create(pipe_model_radius),
         al.uint3(64, 1, 64),
         sign=-1)

pile.build_grids(2 * kernel_radius)
pile.reallocate_kinematics_on_device()
cn.contact_tolerance = particle_radius

# grid
grid_res = al.uint3(int(pipe_radius_grid_span * 2),
                    int(pipe_length_grid_half_span * 2),
                    int(pipe_radius_grid_span * 2))
grid_offset = al.int3(-pipe_radius_grid_span, -pipe_length_grid_half_span,
                      -pipe_radius_grid_span)
cni.grid_res = grid_res
cni.grid_offset = grid_offset
cni.max_num_particles_per_cell = 64
cni.max_num_neighbors_per_particle = 64
cn.set_wrap_length(grid_res.y * kernel_radius)

solver = dp.SolverDf(runner,
                     pile,
                     dp,
                     num_particles,
                     grid_res,
                     enable_surface_tension=False,
                     enable_vorticity=False,
                     graphical=args.display)
solver.particle_radius = particle_radius
solver.num_particles = num_particles
solver.initial_dt = 1e-2
solver.max_dt = 1.0
solver.min_dt = 0
solver.cfl = 0.2

dp.copy_cn()

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

dp.copy_cn()

initial_particle_x = dp.create((num_particles), 3)
initial_particle_x.read_file(args.pos[0])
initial_particle_x.scale(dp.f3(scale_factor, scale_factor, scale_factor))
param0 = np.array([0.0015, 0.006])
adam = AdamOptim(param0, lr=1e-4)

old_filenames = glob.glob('.alcache/*.mp4') + glob.glob('.alcache/*.png')
for filename in old_filenames:
    try:
        os.remove(filename)
    except:
        print("Error while deleting file : ", filename)

wandb.init(project='alluvion')
config = wandb.config
config.dynamic_viscosity = dynamic_viscosity
config.density0 = density0
config.pipe_radius = pipe_radius
config.accelerations = accelerations
config.num_particles = num_particles
config.half_life = approximate_half_life(dynamic_viscosity, density0,
                                         pipe_radius)
config.lambda_factors = lambda_factors
config.kernel_radius = kernel_radius
config.pipe_model_radius = pipe_model_radius
config.precision = str(dp.default_dtype)
config.Re = pipe_radius * pipe_radius * pipe_radius * density0 * density0 * 0.25 / dynamic_viscosity / dynamic_viscosity * accelerations
config.lr = adam.lr
config.initial_dt = solver.initial_dt
config.max_dt = solver.max_dt
config.min_dt = solver.min_dt
config.cfl = solver.cfl
config.density_change_tolerance = solver.density_change_tolerance
config.density_error_tolerance = solver.density_error_tolerance

optimize(dp, solver, adam, param0, initial_particle_x, pipe_radius,
         lambda_factors, density0, accelerations, dynamic_viscosity)
