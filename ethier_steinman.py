import alluvion as al
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser(description='Ethier-Steinman problem')

parser.add_argument('--mode', type=str, default='fill')
args = parser.parse_args()

dp = al.Depot(np.float32)
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
cube_length = 2**-4
# initial_container_mesh.set_box(al.float3(cube_length, cube_length, cube_length), 4)
initial_container_mesh.set_uv_sphere(cube_length * 0.5, 24, 24)
# initial_container_distance = dp.BoxDistance(dp.f3(cube_length, cube_length, cube_length))
initial_container_distance = dp.SphereDistance(cube_length * 0.5)

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
half_fill_length = (cube_length * 0.5) / np.sqrt(3) * fill_factor
box_min = dp.f3(-half_fill_length, -half_fill_length, -half_fill_length)
box_max = dp.f3(half_fill_length, half_fill_length, half_fill_length)
initial_num_particles = dp.Runner.get_fluid_block_num_particles(
    mode=block_mode,
    box_min=box_min,
    box_max=box_max,
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
                     enable_vorticity=False,
                     graphical=True)
if args.mode == 'optimize':
    solver.enable_vorticity = True
    dictator = dp.Solver(runner,
                         pile,
                         dp,
                         max_num_particles,
                         grid_res,
                         enable_surface_tension=False,
                         enable_vorticity=False,
                         graphical=True)


def evaluate_ethier_steinman(dp, runner, solver, dictator, x_filename, a, d,
                             boundary_thickness, dt, solver_box_min,
                             solver_box_max, osampling, viscosity,
                             vorticity_coeff, ts, display_proxy,
                             display_solver):
    # TODO: osampling.reset()
    cn.viscosity = viscosity
    cn.vorticity_coeff = vorticity_coeff
    dp.copy_cn()
    num_copied = dp.create_coated(1, 1, np.uint32)
    num_copied_display = dp.create_coated(1, 1, np.uint32)
    dictator_mask = dp.create_coated(dictator.particle_x.get_shape(), 1,
                                     np.uint32)
    solver_mask = dp.create_coated(solver.particle_x.get_shape(), 1, np.uint32)
    dp.map_graphical_pointers()
    dictator.num_particles = dictator.particle_x.read_file(x_filename)
    boundary_box_min = dp.f3(solver_box_min.x - boundary_thickness,
                             solver_box_min.y - boundary_thickness,
                             solver_box_min.z - boundary_thickness)
    boundary_box_max = dp.f3(solver_box_max.x + boundary_thickness,
                             solver_box_max.y + boundary_thickness,
                             solver_box_max.z + boundary_thickness)

    dictator.t = 0
    dictator.dt = dt
    solver.t = 0
    solver.dt = dt
    solver.max_dt = dt
    solver.min_dt = dt

    dictator.dictate_ethier_steinman(a=a, d=d)
    num_copied.set_zero()
    # runner.launch_copy_kinematics_if_within(dictator.particle_x,
    #                                         dictator.particle_v,
    #                                         solver.particle_x,
    #                                         solver.particle_v,
    #                                         boundary_box_min, boundary_box_max,
    #                                         dictator.num_particles, num_copied)
    runner.launch_copy_kinematics_if_within(dictator.particle_x,
                                            dictator.particle_v,
                                            solver.particle_x,
                                            solver.particle_v, solver_box_min,
                                            solver_box_max,
                                            dictator.num_particles, num_copied)
    num_particles_inner = num_copied.get()[0]
    solver_mask.set_zero()
    solver_mask.set_same(1, num_particles_inner)
    print(num_particles_inner, np.sum(solver_mask.get() == 0x01010101))
    runner.launch_copy_kinematics_if_between(
        dictator.particle_x, dictator.particle_v, solver.particle_x,
        solver.particle_v, boundary_box_min, boundary_box_max, solver_box_min,
        solver_box_max, dictator.num_particles, num_copied)
    solver.num_particles = num_copied.get()[0]

    dp.unmap_graphical_pointers()
    new_solver_x = dp.create_coated(
        dictator.particle_x.get_shape(),
        dictator.particle_x.get_num_primitives_per_element())
    new_solver_v = dp.create_coated(
        dictator.particle_v.get_shape(),
        dictator.particle_v.get_num_primitives_per_element())

    # TODO: while osampling.sampling_cursor < len(ts):
    while True:
        dp.map_graphical_pointers()

        dictator_mask.set_zero()
        dictator.set_mask(dictator_mask, solver_box_min, solver_box_max)
        dictator.move_particles_naive()
        dictator.dictate_ethier_steinman(a=a, d=d)
        dictator.t += dictator.dt

        num_copied.set_zero()
        runner.launch_copy_kinematics_if_within_masked(
            solver.particle_x, solver.particle_v, solver_mask, 0x01010101,
            new_solver_x, new_solver_v, solver_box_min, solver_box_max,
            solver.num_particles, num_copied)
        runner.launch_copy_kinematics_if_within_masked(
            dictator.particle_x, dictator.particle_v, dictator_mask, 1,
            new_solver_x, new_solver_v, solver_box_min, solver_box_max,
            dictator.num_particles, num_copied)
        num_particles_inner = num_copied.get()[0]
        solver_mask.set_zero()
        solver_mask.set_same(1, num_particles_inner)
        runner.launch_copy_kinematics_if_between(
            dictator.particle_x, dictator.particle_v, new_solver_x,
            new_solver_v, boundary_box_min, boundary_box_max, solver_box_min,
            solver_box_max, dictator.num_particles, num_copied)

        num_copied_host = num_copied.get()[0]
        # print('combined', num_copied_host)
        solver.particle_x.set_from(new_solver_x, num_copied_host)
        solver.particle_v.set_from(new_solver_v, num_copied_host)
        solver.num_particles = num_copied_host

        solver.num_particles = num_particles_inner
        solver.step()
        solver.num_particles = num_copied_host
        # print(dictator.t, solver.t, dictator.dt, solver.dt)

        # # DEBUG
        # num_copied_display.set_zero()
        # runner.launch_copy_kinematics_if_between(solver.particle_x, solver.particle_v, display_solver.particle_x, display_solver.particle_v, boundary_box_min, boundary_box_max, solver_box_min, solver_box_max, solver.num_particles, num_copied_display)
        # display_solver.num_particles = num_copied_display.get()[0]

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
        # time.sleep(0.5)


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
                                 box_min=box_min,
                                 box_max=box_max)
dp.unmap_graphical_pointers()

display_proxy.set_camera(dp.f3(0, particle_radius * 60, particle_radius * 80),
                         dp.f3(0, 0, 0))
display_proxy.set_clip_planes(particle_radius * 4, particle_radius * 1e5)
# display_proxy.set_clip_planes(particle_radius * 100, particle_radius * 1e5)
colormap_tex = display_proxy.create_colormap_viridis()

if args.mode == 'fill':
    display_proxy.add_particle_shading_program(solver.particle_x,
                                               particle_normalized_attr,
                                               colormap_tex,
                                               solver.particle_radius, solver)
    # display_proxy.add_particle_shading_program(dictator.particle_x,
    #                                            particle_normalized_attr,
    #                                            colormap_tex,
    #                                            dictator.particle_radius, dictator)
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

frame_id = 0
inited_ethier_steinman = False
solver.next_emission_t = 0.24
finished_filling_frame_id = -1
if args.mode == 'fill':
    while True:
        dp.map_graphical_pointers()
        for frame_interstep in range(10):
            if (finished_filling_frame_id > 0
                    and (frame_id - finished_filling_frame_id > 2000)):
                if not inited_ethier_steinman:
                    solver.particle_v.set_zero()
                    inited_ethier_steinman = True
                    # solver.dt = 1e-4
                    print(solver.dt)
                    solver.particle_x.write_file(
                        f"spherical-x-tilted-{solver.num_particles}.alu",
                        solver.num_particles)
                    solver.dt = particle_radius * 1e-6
                    solver.t = 0
                solver.move_particles_naive()
                solver.dictate_ethier_steinman(a=np.pi / 4 / (cube_length / 8),
                                               d=np.pi / 2 / (cube_length / 8))
                solver.t += solver.dt
            else:
                # solver.emit_single(dp.f3(0, cube_length/2 - particle_radius * 2, 0), dp.f3(particle_radius * 1000, 0 , 0))
                center_original = cube_length / 2 - particle_radius * 4
                v = particle_radius * -500
                solver.emit_circle(center=dp.f3(0, center_original, 0),
                                   v=dp.f3(0, v, 0),
                                   radius=particle_radius * 10,
                                   num_emission=80)
                if (finished_filling_frame_id < 0
                        and solver.num_particles == solver.max_num_particles):
                    solver.particle_x.write_file(
                        f"spherical-x-{solver.num_particles}.alu",
                        solver.num_particles)
                    cn.gravity = dp.f3(gravity_scalar * equicoord,
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
elif args.mode == 'optimize':
    solver_box_half_extent = cube_length * 0.125
    evaluate_ethier_steinman(dp,
                             runner,
                             solver,
                             dictator,
                             x_filename='spherical-x.alu',
                             a=np.pi / 4 / (cube_length / 8),
                             d=np.pi / 2 / (cube_length / 8),
                             boundary_thickness=kernel_radius * 4,
                             dt=particle_radius * 1e-6,
                             solver_box_min=dp.f3(-solver_box_half_extent,
                                                  -solver_box_half_extent,
                                                  -solver_box_half_extent),
                             solver_box_max=dp.f3(solver_box_half_extent,
                                                  solver_box_half_extent,
                                                  solver_box_half_extent),
                             osampling=None,
                             viscosity=0,
                             vorticity_coeff=0.002,
                             ts=None,
                             display_proxy=display_proxy,
                             display_solver=display_solver)
