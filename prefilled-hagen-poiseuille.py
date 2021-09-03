import alluvion as al
import numpy as np
from numpy import linalg as LA
from pathlib import Path
import scipy.special as sc
from sklearn.metrics import mean_squared_error
import glob
import argparse
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
    def __init__(self, dp, cn, pipe_length, pipe_radius, ts):
        self.ts = ts
        self.num_sections = 14
        self.num_rotations = 16
        self.num_rs = 31  # should be odd number to get central velocity
        self.num_samples = self.num_rs * self.num_sections * self.num_rotations

        # r contains 0 but not pipe_radius
        self.rs = pipe_radius * 2 / (self.num_rs + 1) * (
            np.arange(self.num_rs) - self.num_rs // 2)

        self.sample_x = dp.create_coated((self.num_samples), 3)
        self.sample_data3 = dp.create_coated((self.num_samples), 3)
        self.sample_neighbors = dp.create_coated(
            (self.num_samples, dp.cni.max_num_neighbors_per_particle), 4)
        self.sample_num_neighbors = dp.create_coated((self.num_samples), 1,
                                                     np.uint32)
        self.sample_x_host = np.zeros((self.num_samples, 3), dp.default_dtype)
        section_length = pipe_length / self.num_sections
        offset_y_per_rotation = section_length / self.num_rotations
        theta_per_rotation = np.pi * 2 / self.num_rotations
        for i in range(self.num_samples):
            section_id = i // (self.num_rs * self.num_rotations)
            rotation_id = (i // self.num_rs) % (self.num_rotations)
            r_id = i % self.num_rs
            theta = theta_per_rotation * rotation_id
            self.sample_x_host[i] = np.array([
                self.rs[r_id] * np.cos(theta),
                pipe_length * -0.5 + section_length * section_id +
                offset_y_per_rotation * rotation_id,
                self.rs[r_id] * np.sin(theta)
            ], dp.default_dtype)
        self.sample_x.set(self.sample_x_host)
        self.dp = dp
        self.reset()

    def reset(self):
        self.sampling_cursor = 0
        self.vx = np.zeros((len(self.ts), self.num_rs), self.dp.default_dtype)


def pressurize(dp, solver, osampling):
    if solver.t >= osampling.ts[osampling.sampling_cursor]:
        solver.update_particle_neighbors_wrap1()
        runner.launch_make_neighbor_list_wrap1(osampling.sample_x, solver.pid,
                                               solver.pid_length,
                                               osampling.sample_neighbors,
                                               osampling.sample_num_neighbors,
                                               osampling.num_samples)
        runner.launch_sample_fluid(osampling.sample_x, solver.particle_x,
                                   solver.particle_density, solver.particle_v,
                                   osampling.sample_neighbors,
                                   osampling.sample_num_neighbors,
                                   osampling.sample_data3,
                                   osampling.num_samples)
        sample_vx = osampling.sample_data3.get().reshape(
            -1, osampling.num_rs, 3)[:, :, 1]
        osampling.vx[osampling.sampling_cursor] = np.mean(sample_vx, axis=0)
        osampling.sampling_cursor += 1
    solver.step_wrap1()


def evaluate_hagen_poiseuille(dp, solver, initial_x, osampling, param, acc,
                              ts):
    osampling.reset()
    cn.gravity = dp.f3(0, acc, 0)
    cn.viscosity = param[0]
    cn.boundary_viscosity = param[1]
    cn.vorticity_coeff = param[2]
    cn.viscosity_omega = param[3]
    cn.inertia_inverse = 0.5  # recommended by author
    dp.copy_cn()
    dp.map_graphical_pointers()
    solver.particle_x.set_from(initial_x)
    solver.particle_v.set_zero()
    solver.reset_solving_var()
    solver.num_particles = initial_x.get_shape()[0]
    solver.dt = 1e-2
    solver.max_dt = 1e-1
    solver.min_dt = 0
    solver.cfl = 0.2
    solver.t = 0
    step_id = 0
    while osampling.sampling_cursor < len(ts):
        pressurize(dp, solver, osampling)
        step_id += 1
        if (dp.has_display() and step_id % 20 == 0):
            dp.get_display_proxy().draw()
    dp.unmap_graphical_pointers()
    return osampling.vx


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


def simulate(param, dp, solver, osampling, initial_x, pipe_radius, ts,
             density0, accelerations, dynamic_viscosity):
    simulated = np.zeros((len(accelerations), len(ts), osampling.num_rs))
    for acc_id, acc in enumerate(accelerations):
        result = evaluate_hagen_poiseuille(dp, solver, initial_x, osampling,
                                           param, acc, ts)
        simulated[acc_id] = result
    return simulated


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


def optimize(dp, solver, initial_x, pipe_radius, lambda_factors, density0,
             accelerations, dynamic_viscosity):
    approx_half_life = approximate_half_life(dynamic_viscosity, density0,
                                             pipe_radius)
    print('approx_half_life', approx_half_life)
    ts = approx_half_life * lambda_factors
    osampling = OptimSampling(dp, cn, pipe_length, pipe_radius, ts)

    best_loss = np.finfo(np.float64).max
    best_x = None
    x = np.array([8.33481991721471e-8, 5.08257662597844e-7, 1e-8, 1e-8])
    adam = AdamOptim(x, lr=1e-8)
    ground_truth = compute_ground_truth(osampling, pipe_radius, ts, density0,
                                        accelerations, dynamic_viscosity)

    with open('switch', 'w') as f:
        f.write('1')
    for iteration in range(4000):
        with open('switch', 'r') as f:
            if f.read(1) == '0':
                break
        current_x = x
        x, loss, grad, simulated = adam.update(simulate, ground_truth,
                                               mse_loss, x, x * 1e-2, dp,
                                               solver, osampling, initial_x,
                                               pipe_radius, ts, density0,
                                               accelerations,
                                               dynamic_viscosity)
        if (loss < best_loss):
            best_loss = loss
            wandb.summary['best_loss'] = best_loss
            best_x = current_x
            wandb.summary['best_x'] = best_x
        save_result(iteration, ground_truth, simulated, osampling,
                    accelerations)
        param_names = ['vis', 'bvis', 'ω_coeff', 'vis_ω']
        log_object = {'loss': loss, '|∇|': LA.norm(grad)}
        for param_id, param_value in enumerate(current_x):
            log_object[param_names[param_id]] = param_value
            log_object['∇' + param_names[param_id]] = grad[param_id]
        wandb.log(log_object)


dp = al.Depot(np.float32)
cn = dp.cn
cni = dp.cni
if args.display:
    dp.create_display(800, 600, "", False)
display_proxy = dp.get_display_proxy() if args.display else None
runner = dp.Runner()

# Physical constants
density0 = 1000
dynamic_viscosity = 0.001

# Model parameters
scale_factor = 2**-7
particle_radius = 0.25 * scale_factor
kernel_radius = particle_radius * 4
cubical_particle_volume = 8 * particle_radius * particle_radius * particle_radius
volume_relative_to_cube = 0.8
particle_mass = cubical_particle_volume * volume_relative_to_cube * density0

cn.set_cubic_discretization_constants()
cn.set_kernel_radius(kernel_radius)
cn.set_particle_attr(particle_radius, particle_mass, density0)

# Pipe dimension
pipe_radius_grid_span = 13
initial_radius = kernel_radius * pipe_radius_grid_span
pipe_model_radius = 1.01034998e+01 * scale_factor
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
accelerations = np.array([2**-25])

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
                     enable_vorticity=True,
                     graphical=args.display)
solver.particle_radius = particle_radius

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

initial_x = dp.create((num_particles), 3)
initial_x.read_file(args.pos[0])
initial_x.scale(dp.f3(scale_factor, scale_factor, scale_factor))

wandb.init(project='alluvion')
config = wandb.config
config.dynamic_viscosity = dynamic_viscosity
config.density0 = density0
config.pipe_radius = pipe_radius
config.accelerations = accelerations
config.num_particles = num_particles
config.half_life = approximate_half_life(dynamic_viscosity, density0,
                                         pipe_radius)
config.kernel_radius = kernel_radius
config.pipe_model_radius = pipe_model_radius
config.precision = str(dp.default_dtype)
config.Re = pipe_radius * pipe_radius * pipe_radius * density0 * density0 * 0.25 / dynamic_viscosity / dynamic_viscosity * accelerations

optimize(dp, solver, initial_x, pipe_radius, lambda_factors, density0,
         accelerations, dynamic_viscosity)
