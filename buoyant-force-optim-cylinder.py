import alluvion as al
import numpy as np

from util import Unit

dp = al.Depot(np.float32)
cn = dp.cn
cni = dp.cni
dp.create_display(800, 600, "", False)
display_proxy = dp.get_display_proxy()
runner = dp.Runner()

particle_radius = 0.25
kernel_radius = 1.0
density0 = 1.0
cubical_particle_volume = 8 * particle_radius * particle_radius * particle_radius
volume_relative_to_cube = 0.8
particle_mass = cubical_particle_volume * volume_relative_to_cube * density0

gravity = dp.f3(0, -1, 0)

unit = Unit(real_kernel_radius=0.005,
            real_density0=1000,
            real_gravity=-9.80665)

cn.set_cubic_discretization_constants()
cn.set_kernel_radius(kernel_radius)
cn.set_particle_attr(particle_radius, particle_mass, density0)
cn.gravity = gravity
cn.viscosity, cn.boundary_viscosity = unit.from_real_kinematic_viscosity(
    np.array([2.049e-6, 6.532e-6]))

# rigids
max_num_contacts = 512
pile = dp.Pile(dp, runner, max_num_contacts)

container_radius = unit.from_real_length(20e-3)
container_height = unit.from_real_length(80e-3)
fluid_y_range = (-container_height / 2, container_height / 2)
container_map_radial_size = 32
pile.add(dp.CapsuleDistance.create(container_radius, container_height),
         al.uint3(
             container_map_radial_size,
             int(container_map_radial_size *
                 (container_height + container_radius * 2) / container_radius /
                 2), container_map_radial_size),
         sign=-1,
         collision_mesh=al.Mesh(),
         mass=0,
         restitution=0,
         friction=0,
         inertia_tensor=dp.f3(1, 1, 1),
         x=dp.f3(0, 0, 0),
         q=dp.f4(0, 0, 0, 1),
         display_mesh=al.Mesh())

current_inset = 0.3984

cylinder_radius = unit.from_real_length(3.0088549658278843e-3)
cylinder_height = unit.from_real_length(38.5e-3)
cylinder_comy = unit.from_real_length(-8.852102803738316e-3)
cylinder_volume = cylinder_radius * cylinder_radius * np.pi * cylinder_height
cylinder_neutral_buoyant_force = -cylinder_volume * density0 * gravity.y
cylinder_mesh = al.Mesh()
cylinder_mesh.set_cylinder(cylinder_radius, cylinder_height, 24, 24)
cylinder_mesh.translate(dp.f3(0, -cylinder_comy, 0))
cylinder_inertia = unit.from_real_moment_of_inertia(
    dp.f3(7.911343969145678e-8, 2.944622178863632e-8, 7.911343969145678e-8))
cylinder_map_radial_size = 24
cylinder_map_dim = al.uint3(
    cylinder_map_radial_size,
    int(cylinder_map_radial_size * cylinder_height / cylinder_radius / 2),
    cylinder_map_radial_size)

pile.add(dp.CylinderDistance.create(cylinder_radius - current_inset,
                                    cylinder_height - current_inset * 2,
                                    cylinder_comy),
         cylinder_map_dim,
         sign=1,
         collision_mesh=cylinder_mesh,
         mass=0,
         restitution=0,
         friction=0,
         inertia_tensor=cylinder_inertia,
         x=dp.f3(0, container_height * 0.5 + cylinder_height * 0.6, 0.0),
         q=dp.f4(0, 0, 0, 1),
         display_mesh=cylinder_mesh)

pile.build_grids(2 * kernel_radius)
pile.reallocate_kinematics_on_device()
pile.set_gravity(gravity)
cn.contact_tolerance = particle_radius

num_particles = dp.Runner.get_fluid_cylinder_num_particles(
    radius=container_radius,
    y_min=fluid_y_range[0],
    y_max=fluid_y_range[1],
    particle_radius=particle_radius)

grid_res = al.uint3(128, 64, 128)
grid_offset = al.int3(-64, -32, -64)

cni.grid_res = grid_res
cni.grid_offset = grid_offset
cni.max_num_particles_per_cell = 64
cni.max_num_neighbors_per_particle = 64

solver = dp.SolverI(runner,
                    pile,
                    dp,
                    num_particles,
                    enable_surface_tension=False,
                    enable_vorticity=False,
                    graphical=True)
particle_normalized_attr = dp.create_graphical((num_particles), 1)

solver.max_dt = unit.from_real_time(0.01 * unit.rl)
solver.initial_dt = solver.max_dt * 0.01
solver.min_dt = 0
solver.cfl = 0.02
solver.num_particles = num_particles

dp.copy_cn()

dp.map_graphical_pointers()

runner.launch_create_fluid_cylinder(solver.particle_x,
                                    solver.num_particles,
                                    offset=0,
                                    radius=container_radius,
                                    particle_radius=particle_radius,
                                    y_min=fluid_y_range[0],
                                    y_max=fluid_y_range[1])
dp.unmap_graphical_pointers()
display_proxy.set_camera(unit.from_real_length(dp.f3(0, 0.06, 0.4)),
                         unit.from_real_length(dp.f3(0, 0.06, 0)))
colormap_tex = display_proxy.create_colormap_viridis()

display_proxy.add_particle_shading_program(solver.particle_x,
                                           particle_normalized_attr,
                                           colormap_tex,
                                           solver.particle_radius, solver)
display_proxy.add_pile_shading_program(pile)

target_y = container_height * -0.5 - container_radius + cylinder_height * 0.6
target_reached = False
num_stabilized = 0
stabilization_target = 10
buoyant_force_cursor = 0
buoyant_force_buffer = np.zeros(10000, dp.default_dtype)
while True:
    dp.map_graphical_pointers()
    for frame_interstep in range(10):
        if (pile.x[1].y > target_y):
            pile.v[1] = unit.from_real_velocity(dp.f3(0, -0.05, 0))
        elif not target_reached:
            target_reached = True
            pile.v[1] = dp.f3(0, 0, 0)
        solver.step()
        if num_stabilized >= stabilization_target:
            if buoyant_force_cursor == 0:
                for further_stabilization_step in range(10000):
                    solver.step()
            buoyant_force = runner.sum(solver.particle_force,
                                       solver.particle_force.get_shape()[1],
                                       solver.particle_force.get_shape()[1]).y
            buoyant_force_buffer[buoyant_force_cursor] = buoyant_force
            buoyant_force_cursor += 1
            if (buoyant_force_cursor >= len(buoyant_force_buffer)):
                mean_buoyant_force = np.mean(buoyant_force_buffer)
                print(
                    f'Inset: {current_inset} {mean_buoyant_force}/{cylinder_neutral_buoyant_force}={mean_buoyant_force/cylinder_neutral_buoyant_force}'
                )
                print(buoyant_force_buffer[[1, 5, 500, 600, 2000, 9000, 9999]])
                # reset
                current_inset += unit.from_real_length(5e-7)
                num_stabilized = 0
                stabilization_target = 5
                buoyant_force_cursor = 0
                pile.replace(1,
                             dp.CylinderDistance.create(
                                 cylinder_radius - current_inset,
                                 cylinder_height - current_inset * 2,
                                 cylinder_comy),
                             cylinder_map_dim,
                             sign=1,
                             collision_mesh=cylinder_mesh,
                             mass=0,
                             restitution=0,
                             friction=0,
                             inertia_tensor=cylinder_inertia,
                             x=pile.x[1],
                             q=dp.f4(0, 0, 0, 1),
                             display_mesh=cylinder_mesh)
                pile.build_grids(2 * kernel_radius)

    if target_reached:
        if num_stabilized >= stabilization_target:
            pass
        else:
            if (int(unit.to_real_time(solver.t) * 1000) % 500 == 0):
                solver.particle_v.set_zero()
                solver.reset_solving_var()
                num_stabilized += 1
                print("num_stabilized", num_stabilized)

    solver.normalize(solver.particle_v, particle_normalized_attr, 0, 2)
    dp.unmap_graphical_pointers()
    display_proxy.draw()
