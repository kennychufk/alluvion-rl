import alluvion as al
import numpy as np

dp = al.Depot(np.float32)
dp.set_device(0)
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
gravity = dp.f3(0, -9.81, 0)

cn.set_cubic_discretization_constants()
cn.set_kernel_radius(kernel_radius)
cn.set_particle_attr(particle_radius, particle_mass, density0)
cn.boundary_epsilon = 1e-9
cn.gravity = gravity
cn.viscosity = 1.37916076e-05
cn.boundary_viscosity = 9.96210578e-06

# rigids
max_num_contacts = 512
pile = dp.Pile(dp, max_num_contacts)

container_radius = 30e-3
container_height = 90e-3
cylinder_container_mesh = al.Mesh()
cylinder_container_mesh.set_cylinder(container_radius,
                                     container_height + container_radius * 2,
                                     24, 24)
fluid_y_range = (-container_height / 2, container_height / 2 * 0.45)
capsule_distance = dp.CapsuleDistance(container_radius, container_height)
pile.add(capsule_distance,
         al.uint3(32, 96, 32),
         sign=-1,
         thickness=0,
         collision_mesh=cylinder_container_mesh,
         mass=0,
         restitution=0,
         friction=0,
         inertia_tensor=dp.f3(1, 1, 1),
         x=dp.f3(0, 0, 0),
         q=dp.f4(0, 0, 0, 1),
         display_mesh=al.Mesh())

inset_factor = 1.71

cylinder_radius = 3.00885e-3
cylinder_height = 38.5e-3
cylinder_comy = -8.8521e-3
cylinder_mass = 1.06e-3
cylinder_mesh = al.Mesh()
cylinder_mesh.set_cylinder(cylinder_radius, cylinder_height, 24, 24)
cylinder_mesh.translate(dp.f3(0, -cylinder_comy, 0))
cylinder_distance = dp.CylinderDistance(cylinder_radius, cylinder_height,
                                        cylinder_comy)
pile.add(cylinder_distance,
         al.uint3(20, 128, 20),
         sign=1,
         thickness=-particle_radius * inset_factor,
         collision_mesh=cylinder_mesh,
         mass=cylinder_mass,
         restitution=0,
         friction=0,
         inertia_tensor=dp.f3(7.91134e-8, 2.94462e-9, 7.91134e-8),
         x=dp.f3(
             0, 30e-3, 0.0),
         q=dp.f4(np.sqrt(2)*0.5, 0, 0, np.sqrt(2)*0.5),
         display_mesh=cylinder_mesh)

cylinder_density = cylinder_mass / (cylinder_radius * cylinder_radius * np.pi *
                                    cylinder_height)
print(cylinder_density)

pile.build_grids(4 * kernel_radius)
pile.reallocate_kinematics_on_device()
pile.set_gravity(gravity)
cn.contact_tolerance = particle_radius

max_num_particles = dp.Runner.get_fluid_cylinder_num_particles(
    radius=container_radius,
    y_min=fluid_y_range[0],
    y_max=fluid_y_range[1],
    particle_radius=particle_radius)
grid_res = al.uint3(128, 128, 128)
grid_offset = al.int3(-64, -64, -64)

cni.grid_res = grid_res
cni.grid_offset = grid_offset
cni.max_num_particles_per_cell = 64
cni.max_num_neighbors_per_particle = 64

solver = dp.SolverDf(runner, pile, dp, max_num_particles, grid_res,
                     enable_surface_tension = False, enable_vorticity = False, graphical = True)
particle_normalized_attr = dp.create_graphical((max_num_particles), 1)

solver.dt = 1e-3
solver.max_dt = 5e-5
solver.min_dt = 0.0
solver.cfl = 0.04
solver.particle_radius = particle_radius
solver.num_particles = max_num_particles

dp.copy_cn()

dp.map_graphical_pointers()
#2 ^-11
runner.launch_create_fluid_cylinder(256,
                                    solver.particle_x,
                                    solver.num_particles,
                                    offset=0,
                                    radius=container_radius,
                                    y_min=fluid_y_range[0],
                                    y_max=fluid_y_range[1])
dp.unmap_graphical_pointers()
display_proxy.set_camera(dp.f3(0, 0.06, 0.4), dp.f3(0, 0.06, 0))
colormap_tex = display_proxy.create_colormap_viridis()

# display_proxy.add_map_graphical_pointers(dp)
# display_proxy.add_step(solver, 10)
# display_proxy.add_normalize(solver, solver.particle_v,
#                             particle_normalized_attr, 0, 2)
# display_proxy.add_unmap_graphical_pointers(dp)
# display_proxy.add_particle_shading_program(solver.particle_x,
#                                            particle_normalized_attr,
#                                            colormap_tex,
#                                            solver.particle_radius, solver)
# display_proxy.add_pile_shading_program(pile)
#
# display_proxy.run()

display_proxy.add_particle_shading_program(solver.particle_x,
                                           particle_normalized_attr,
                                           colormap_tex,
                                           solver.particle_radius, solver)
display_proxy.add_pile_shading_program(pile)

while True:
    dp.map_graphical_pointers()
    for frame_interstep in range(10):
        buoyant_force = runner.sum(solver.particle_force,
                                   solver.particle_force.get_shape()[1],
                                   solver.particle_force.get_shape()[1]).y
        print(buoyant_force, pile.x[1], pile.v[1], pile.q[1], pile.omega[1])
        solver.step()

    solver.normalize(solver.particle_v, particle_normalized_attr, 0, 2)
    dp.unmap_graphical_pointers()
    display_proxy.draw()
