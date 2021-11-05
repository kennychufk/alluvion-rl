import scipy.special as sc
import numpy as np
from numpy import linalg as LA


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


def pressurize(dp, solver, osampling):
    runner = solver.runner
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
