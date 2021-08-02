from optim import AdamOptim
import alluvion as al
import numpy as np
from sklearn.metrics import mean_squared_error
import argparse
import time


class EthierSteinmanParam:
    def __init__(self, a, d, kinematic_viscosity):
        self.a = a
        self.d = d
        self.kinematic_viscosity = kinematic_viscosity


def ethier_steinman_ans(es_param, osampling):
    a = es_param.a
    d = es_param.d
    kinematic_viscosity = es_param.kinematic_viscosity
    ans = np.zeros_like(osampling.v_host)
    for t_id, t in enumerate(osampling.ts):
        temporal_term = -a * np.exp(-kinematic_viscosity * d * d * t)
        for x_id, sample_x in enumerate(osampling.sample_x_host):
            exp_ax = np.exp(a * sample_x[0])
            exp_ay = np.exp(a * sample_x[1])
            exp_az = np.exp(a * sample_x[2])
            ay_dz = a * sample_x[1] + d * sample_x[2]
            ax_dy = a * sample_x[0] + d * sample_x[1]
            az_dx = a * sample_x[2] + d * sample_x[0]
            ans[t_id, x_id] = temporal_term * np.array([
                exp_ax * np.sin(ay_dz) + exp_az * np.cos(ax_dy),
                exp_ay * np.sin(az_dx) + exp_ax * np.cos(ay_dz),
                exp_az * np.sin(ax_dy) + exp_ay * np.cos(az_dx)
            ])
    return ans


def evaluate_ethier_steinman(viscosity, vorticity_coeff, viscosity_omega, dp,
                             runner, solver, x_filename, es_param, dt,
                             solver_box_min, solver_box_max, osampling,
                             display_proxy, display_solver):
    osampling.reset()
    dp.cn.viscosity = viscosity
    dp.cn.vorticity_coeff = vorticity_coeff
    dp.cn.viscosity_omega = viscosity_omega
    dp.copy_cn()
    num_copied_display = dp.create_coated(1, 1, np.uint32)

    solver.t = 0
    solver.dt = dt
    solver.max_dt = dt
    solver.min_dt = dt
    world_min = dp.f3((dp.cni.grid_offset.x + 1) * dp.cn.kernel_radius,
                      (dp.cni.grid_offset.y + 1) * dp.cn.kernel_radius,
                      (dp.cni.grid_offset.z + 1) * dp.cn.kernel_radius)
    world_max = dp.f3(
        (dp.cni.grid_res.x + dp.cni.grid_offset.x - 1) * dp.cn.kernel_radius,
        (dp.cni.grid_res.y + dp.cni.grid_offset.y - 1) * dp.cn.kernel_radius,
        (dp.cni.grid_res.z + dp.cni.grid_offset.z - 1) * dp.cn.kernel_radius)

    dp.map_graphical_pointers()
    solver.num_particles = solver.particle_x.read_file(x_filename)
    solver.dictate_ethier_steinman(
        a=es_param.a,
        d=es_param.d,
        kinematic_viscosity=es_param.kinematic_viscosity)
    dp.unmap_graphical_pointers()
    new_solver_x = dp.create_coated(
        solver.particle_x.get_shape(),
        solver.particle_x.get_num_primitives_per_element())
    new_solver_v = dp.create_coated(
        solver.particle_v.get_shape(),
        solver.particle_v.get_num_primitives_per_element())
    num_copied = dp.create_coated(1, 1, np.uint32)

    while osampling.sampling_cursor < len(osampling.ts):
        dp.map_graphical_pointers()

        if solver.t >= osampling.ts[osampling.sampling_cursor]:
            solver.pid_length.set_zero()
            runner.launch_update_particle_grid(256, solver.particle_x,
                                               solver.pid, solver.pid_length,
                                               solver.num_particles)
            runner.launch_make_neighbor_list_wrap1(
                256, osampling.sample_x, solver.pid, solver.pid_length,
                osampling.sample_neighbors, osampling.sample_num_neighbors,
                osampling.num_samples)
            runner.launch_compute_density(
                256, solver.particle_x, solver.particle_neighbors,
                solver.particle_num_neighbors, solver.particle_density,
                solver.particle_boundary_xj, solver.particle_boundary_volume,
                solver.num_particles)
            runner.launch_sample_fluid(
                256, osampling.sample_x, solver.particle_x,
                solver.particle_density, solver.particle_v,
                osampling.sample_neighbors, osampling.sample_num_neighbors,
                osampling.sample_data3, osampling.num_samples)
            osampling.v_host[
                osampling.sampling_cursor] = osampling.sample_data3.get()
            osampling.sampling_cursor += 1

        solver.step()
        solver.dictate_ethier_steinman(
            a=es_param.a,
            d=es_param.d,
            kinematic_viscosity=es_param.kinematic_viscosity,
            exclusion_min=solver_box_min,
            exclusion_max=solver_box_max)
        num_copied.set_zero()
        runner.launch_copy_kinematics_if_within(
            solver.particle_x, solver.particle_v, new_solver_x, new_solver_v,
            world_min, world_max, solver.num_particles, num_copied)
        num_copied_host = num_copied.get()[0]
        if (num_copied_host != solver.num_particles):
            solver.particle_x.set_from(new_solver_x, num_copied_host)
            solver.particle_v.set_from(new_solver_v, num_copied_host)
            solver.num_particles = num_copied_host
        # DEBUG
        num_copied_display.set_zero()
        runner.launch_copy_kinematics_if_within(
            solver.particle_x, solver.particle_v, display_solver.particle_x,
            display_solver.particle_v, solver_box_min, solver_box_max,
            solver.num_particles, num_copied_display)
        display_solver.num_particles = num_copied_display.get()[0]

        # TODO: solver.normalize(solver.particle_v, particle_normalized_attr, 0, 0.5)
        dp.unmap_graphical_pointers()
        display_proxy.draw()
    return osampling.v_host


def compute_ground_truth_and_simulate(param, dp, runner, solver, x_filename,
                                      es_param, dt, solver_box_min,
                                      solver_box_max, osampling, display_proxy,
                                      display_solver):
    ans = ethier_steinman_ans(es_param, osampling)
    simulated = evaluate_ethier_steinman(*param, dp, runner, solver,
                                         x_filename, es_param, dt,
                                         solver_box_min, solver_box_max,
                                         osampling, display_proxy,
                                         display_solver)
    return ans, simulated


def evaluate_loss(param, dp, runner, solver, x_filename, es_param, dt,
                  solver_box_min, solver_box_max, osampling, display_proxy,
                  display_solver):
    ans, simulated = compute_ground_truth_and_simulate(
        param, dp, runner, solver, x_filename, es_param, dt, solver_box_min,
        solver_box_max, osampling, display_proxy, display_solver)
    return mean_squared_error(ans.reshape(-1, 3), simulated.reshape(-1, 3))


def optimize(dp, runner, solver, x_filename, es_param, dt, solver_box_min,
             solver_box_max, osampling, display_proxy, display_solver):
    best_loss = np.finfo(np.float64).max
    best_x = None
    x = np.array([3.61355447e-06, 0.01, 0.1])
    adam = AdamOptim(x, lr=1e-8)

    for iteration in range(30):
        current_x = x
        x, loss, grad = adam.update(evaluate_loss, x, x * 1e-2, dp, runner,
                                    solver, x_filename, es_param, dt,
                                    solver_box_min, solver_box_max, osampling,
                                    display_proxy, display_solver)
        if (loss < best_loss):
            best_loss = loss
            best_x = current_x
        print(current_x, x, loss, grad)
        print('best x', best_x)


def fill_until_rest(dp, solver, sphere_radius, particle_normalized_attr):
    frame_id = 0
    solver.next_emission_t = 0.24
    finished_filling_frame_id = -1
    while True:
        dp.map_graphical_pointers()
        for frame_interstep in range(10):
            if (finished_filling_frame_id > 0
                    and (frame_id - finished_filling_frame_id > 2000)):
                solver.particle_v.set_zero()
                solver.particle_x.write_file(
                    f"spherical-x-tilted-{solver.num_particles}.alu",
                    solver.num_particles)
                return
            else:
                center_original = sphere_radius - dp.cn.particle_radius * 4
                v = dp.cn.particle_radius * -500
                solver.emit_circle(center=dp.f3(0, center_original, 0),
                                   v=dp.f3(0, v, 0),
                                   radius=dp.cn.particle_radius * 10,
                                   num_emission=80)
                if (finished_filling_frame_id < 0
                        and solver.num_particles == solver.max_num_particles):
                    solver.particle_x.write_file(
                        f"spherical-x-{solver.num_particles}.alu",
                        solver.num_particles)
                    dp.cn.gravity = dp.f3(gravity_scalar * equicoord,
                                          gravity_scalar * equicoord,
                                          gravity_scalar * equicoord)
                    dp.copy_cn()
                    finished_filling_frame_id = frame_id
                solver.step()
        print(solver.t, frame_id)
        print("num_particles", solver.num_particles, solver.max_num_particles)
        solver.normalize(solver.particle_v, particle_normalized_attr, 0, 0.05)
        dp.unmap_graphical_pointers()
        display_proxy.draw()
        frame_id += 1


class OptimSampling:
    def __init__(self, dp, solver_box_min, solver_box_max, ts):
        self.ts = ts
        self.dp = dp
        self.num_samples_per_dim = 31
        self.num_samples = self.num_samples_per_dim * self.num_samples_per_dim * self.num_samples_per_dim
        self.sample_x = dp.create_coated((self.num_samples), 3)
        self.sample_data3 = dp.create_coated((self.num_samples), 3)
        self.sample_neighbors = dp.create_coated(
            (self.num_samples, dp.cni.max_num_neighbors_per_particle), 4)
        self.sample_num_neighbors = dp.create_coated((self.num_samples), 1,
                                                     np.uint32)

        self.sample_x_host = np.zeros((self.num_samples, 3), dp.default_dtype)

        solver_box_size = dp.f3(solver_box_max.x - solver_box_min.x,
                                solver_box_max.y - solver_box_min.y,
                                solver_box_max.z - solver_box_min.z)
        for i in range(self.num_samples):
            z_id = i % self.num_samples_per_dim
            y_id = i % (self.num_samples_per_dim *
                        self.num_samples_per_dim) // self.num_samples_per_dim
            x_id = i // (self.num_samples_per_dim * self.num_samples_per_dim)
            self.sample_x_host[i] = np.array([
                solver_box_min.x + solver_box_size.x /
                (self.num_samples_per_dim - 1) * x_id, solver_box_min.y +
                solver_box_size.y / (self.num_samples_per_dim - 1) * y_id,
                solver_box_min.z + solver_box_size.z /
                (self.num_samples_per_dim - 1) * z_id
            ])
        self.sample_x.set(self.sample_x_host)
        self.reset()

    def reset(self):
        self.sampling_cursor = 0
        self.v_host = np.zeros((len(self.ts), self.num_samples, 3),
                               self.dp.default_dtype)


parser = argparse.ArgumentParser(description='Ethier-Steinman problem')

parser.add_argument('--mode', type=str, default='fill')
args = parser.parse_args()

dp = al.Depot(np.float64)
cn = dp.cn
cni = dp.cni
dp.create_display(800, 600, "", False)
display_proxy = dp.get_display_proxy()
runner = dp.Runner()
particle_radius = 2**-11
kernel_radius = particle_radius * 4
density0 = 1000.0
cubical_particle_volume = 8 * particle_radius * particle_radius * particle_radius
volume_relative_to_cube = 0.82
particle_mass = cubical_particle_volume * volume_relative_to_cube * density0
dt = 2e-3
gravity_scalar = -9.81
equicoord = np.sqrt(1 / 3)

cn.set_cubic_discretization_constants()
cn.set_kernel_radius(kernel_radius)
cn.set_particle_attr(particle_radius, particle_mass, density0)
cn.inertia_inverse = 0.5
cn.vorticity_coeff = 0.01
cn.viscosity_omega = 0.1

if args.mode == 'fill':
    cn.gravity = dp.f3(0, gravity_scalar, 0)
    cn.viscosity = 3.54008928e-06
    cn.boundary_viscosity = 6.71368218e-06
else:
    cn.gravity = dp.f3(0, 0, 0)

# rigids
max_num_contacts = 512
pile = dp.Pile(dp, max_num_contacts)
initial_container_mesh = al.Mesh()
sphere_radius = 2**-5
initial_container_mesh.set_uv_sphere(sphere_radius, 24, 24)
initial_container_distance = dp.SphereDistance(sphere_radius)

if args.mode == 'fill':
    pile.add(initial_container_distance,
             al.uint3(64, 64, 64),
             sign=-1,
             thickness=0,
             collision_mesh=initial_container_mesh,
             mass=0,
             restitution=1,
             friction=0,
             inertia_tensor=dp.f3(1, 1, 1),
             x=dp.f3(0, 0, 0),
             q=dp.f4(0, 0, 0, 1),
             display_mesh=al.Mesh())

pile.build_grids(4 * kernel_radius)
pile.reallocate_kinematics_on_device()
pile.set_gravity(cn.gravity)
cn.contact_tolerance = particle_radius

block_mode = 0
fill_factor = 0.99
half_fill_length = (sphere_radius) / np.sqrt(3) * fill_factor
fill_box_min = dp.f3(-half_fill_length, -half_fill_length, -half_fill_length)
fill_box_max = dp.f3(half_fill_length, half_fill_length, half_fill_length)
initial_num_particles = dp.Runner.get_fluid_block_num_particles(
    mode=block_mode,
    box_min=fill_box_min,
    box_max=fill_box_max,
    particle_radius=particle_radius)
max_num_particles = int(initial_num_particles * 3.3)
grid_res = al.uint3(128, 128, 128)
grid_offset = al.int3(-64, -64, -64)

cni.grid_res = grid_res
cni.grid_offset = grid_offset
cni.max_num_particles_per_cell = 64
cni.max_num_neighbors_per_particle = 64

solver = dp.SolverDf(runner,
                     pile,
                     dp,
                     max_num_particles,
                     grid_res,
                     enable_surface_tension=False,
                     enable_vorticity=True,
                     graphical=True)

particle_normalized_attr = dp.create_graphical((max_num_particles), 1)
solver.num_particles = initial_num_particles
solver.dt = dt
solver.max_dt = particle_radius * 0.08
solver.min_dt = 0.0
solver.cfl = 0.08

dp.copy_cn()

dp.map_graphical_pointers()
runner.launch_create_fluid_block(256,
                                 solver.particle_x,
                                 initial_num_particles,
                                 offset=0,
                                 mode=block_mode,
                                 box_min=fill_box_min,
                                 box_max=fill_box_max)
dp.unmap_graphical_pointers()

display_proxy.set_camera(
    al.float3(0, particle_radius * 60, particle_radius * 80),
    al.float3(0, 0, 0))
display_proxy.set_clip_planes(particle_radius * 4, particle_radius * 1e5)
# display_proxy.set_clip_planes(particle_radius * 100, particle_radius * 1e5)
colormap_tex = display_proxy.create_colormap_viridis()

if args.mode == 'fill':
    display_proxy.add_particle_shading_program(solver.particle_x,
                                               particle_normalized_attr,
                                               colormap_tex,
                                               solver.particle_radius, solver)
    display_proxy.add_pile_shading_program(pile)
else:
    display_solver = dp.Solver(runner,
                               pile,
                               dp,
                               max_num_particles,
                               grid_res,
                               enable_surface_tension=False,
                               enable_vorticity=False,
                               graphical=True)
    # display_proxy.add_particle_shading_program(solver.particle_x,
    #                                            particle_normalized_attr,
    #                                            colormap_tex,
    #                                            solver.particle_radius, solver)
    display_proxy.add_particle_shading_program(display_solver.particle_x,
                                               particle_normalized_attr,
                                               colormap_tex,
                                               display_solver.particle_radius,
                                               display_solver)

if args.mode == 'fill':
    fill_until_rest(dp, solver, sphere_radius, particle_normalized_attr)
elif args.mode == 'optimize':
    solver_box_half_extent = sphere_radius * 0.3
    solver_box_min = dp.f3(-solver_box_half_extent, -solver_box_half_extent,
                           -solver_box_half_extent)
    solver_box_max = dp.f3(solver_box_half_extent, solver_box_half_extent,
                           solver_box_half_extent)
    es_param = EthierSteinmanParam(a=np.pi / 4 / (sphere_radius),
                                   d=np.pi / 2 / (sphere_radius),
                                   kinematic_viscosity=20e-6)
    dt = particle_radius * 1e-6
    osampling = OptimSampling(dp, solver_box_min, solver_box_max,
                              np.array([dt * 500]))
    optimize(dp, runner, solver, 'spherical-x.alu', es_param, dt,
             solver_box_min, solver_box_max, osampling, display_proxy,
             display_solver)
