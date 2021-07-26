import alluvion as al
import numpy as np

dp = al.Depot(np.float32)
cn = dp.cn
cni = dp.cni
dp.create_display(800, 600, "", False)
display_proxy = dp.get_display_proxy()
runner = dp.Runner()
particle_radius = 0.025
kernel_radius = particle_radius * 4
density0 = 1000.0
particle_mass = 0.1
dt = 2e-3
gravity = dp.f3(0, -9.81, 0)

cn.set_cubic_discretization_constants()
cn.set_kernel_radius(kernel_radius)
cn.set_particle_attr(particle_radius, particle_mass, density0)
cn.gravity = gravity
cn.viscosity = 0.001

# rigids
max_num_contacts = 512
pile = dp.Pile(dp, max_num_contacts)
cube_mesh = al.Mesh()
cube_mesh.set_box(al.float3(4, 3, 1.5), 4)
box_distance = dp.BoxDistance(dp.f3(4, 3, 1.5))
pile.add(box_distance,
         al.uint3(80, 60, 30),
         sign=-1,
         thickness=0,
         collision_mesh=cube_mesh,
         mass=0,
         restitution=1,
         friction=0,
         inertia_tensor=dp.f3(1, 1, 1),
         x=dp.f3(0, 1.5, 0),
         q=dp.f4(0, 0, 0, 1),
         display_mesh=al.Mesh())

inset_factor = 0
sphere_radius = 0.1
sphere_distance = dp.SphereDistance(sphere_radius -
                                    particle_radius * inset_factor)
sphere_mesh = al.Mesh()
sphere_mesh.set_uv_sphere(sphere_radius, 24, 24)
sphere_mass = density0 * sphere_radius * sphere_radius * sphere_radius * np.pi * 4 / 3 * 0.7
pile.add(sphere_distance,
         al.uint3(64, 64, 64),
         sign=1,
         thickness=0,
         collision_mesh=sphere_mesh,
         mass=sphere_mass,
         restitution=0,
         friction=0,
         inertia_tensor=dp.f3(1, 1, 1),
         x=dp.f3(0, 0.4, 0),
         q=dp.f4(0, 0, 0, 1),
         display_mesh=sphere_mesh)

pile.build_grids(4 * kernel_radius)
pile.reallocate_kinematics_on_device()
pile.set_gravity(gravity)
cn.contact_tolerance = particle_radius

block_mode = 0
box_min = dp.f3(-1.95, 0, -0.5)
box_max = dp.f3(-0.95, 1, 0.5)
num_particles = dp.Runner.get_fluid_block_num_particles(
    mode=block_mode,
    box_min=box_min,
    box_max=box_max,
    particle_radius=particle_radius)
grid_res = al.uint3(128, 128, 128)
grid_offset = al.int3(-64, -64, -64)

cni.grid_res = grid_res
cni.grid_offset = grid_offset
cni.max_num_particles_per_cell = 64
cni.max_num_neighbors_per_particle = 64

solver = dp.SolverDf(runner, pile, dp, num_particles, grid_res, False, False,
                     True)
particle_normalized_attr = dp.create_graphical((num_particles), 1)
solver.num_particles = num_particles
solver.dt = dt
solver.max_dt = 0.005
solver.min_dt = 0.0001
solver.cfl = 0.4

dp.copy_cn()

dp.map_graphical_pointers()
runner.launch_create_fluid_block(256,
                                 solver.particle_x,
                                 num_particles,
                                 offset=0,
                                 mode=block_mode,
                                 box_min=box_min,
                                 box_max=box_max)
dp.unmap_graphical_pointers()

display_proxy.set_camera(al.float3(0, 6, 6), al.float3(0, -0.2, 0))
colormap_tex = display_proxy.create_colormap_viridis()

display_proxy.add_map_graphical_pointers(dp)
display_proxy.add_step(solver, 10)
display_proxy.add_normalize(solver, solver.particle_v,
                            particle_normalized_attr, 0, 2)
# display_proxy.add_normalize(solver, solver.particle_kappa_v, particle_normalized_attr, -0.002, 0)
display_proxy.add_unmap_graphical_pointers(dp)
display_proxy.add_particle_shading_program(solver.particle_x,
                                           particle_normalized_attr,
                                           colormap_tex,
                                           solver.particle_radius, solver)
display_proxy.add_pile_shading_program(pile)

display_proxy.run()
