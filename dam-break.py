import math
import time

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

# real_kernel_radius = 0.0025
unit = Unit(real_kernel_radius=0.005,
            real_density0=1000,
            real_gravity=-9.80665)

cn.set_cubic_discretization_constants()
cn.set_kernel_radius(kernel_radius)
cn.set_particle_attr(particle_radius, particle_mass, density0)
cn.gravity = gravity
cn.viscosity, cn.boundary_viscosity = unit.from_real_kinematic_viscosity(
    np.array([2.049e-6, 6.532e-6]))

cn.inertia_inverse = 0.5
cn.vorticity_coeff = 0.01
cn.viscosity_omega = 0.1

# rigids
max_num_contacts = 512
pile = dp.Pile(dp, runner, max_num_contacts)
cube_mesh = al.Mesh()
real_outset = unit.rl * 0.5  # NOTE: unstable when equal to unit.rl
container_dim = unit.from_real_length(al.float3(0.4, 0.3, 0.15) + real_outset)
cube_mesh.set_box(container_dim, 4)
container_distance = dp.BoxDistance.create(
    unit.from_real_length(dp.f3(0.4, 0.3, 0.15)),
    outset=unit.from_real_length(real_outset))
pile.add(container_distance,
         al.uint3(80, 60, 30),
         sign=-1,
         thickness=0,
         collision_mesh=cube_mesh,
         mass=0,
         restitution=1,
         friction=0,
         inertia_tensor=dp.f3(1, 1, 1),
         x=unit.from_real_length(dp.f3(0, 0.15, 0)),
         q=dp.f4(0, 0, 0, 1),
         display_mesh=al.Mesh())

inset_factor = 0
sphere_radius = unit.from_real_length(0.01)
sphere_mesh = al.Mesh()
sphere_mesh.set_uv_sphere(sphere_radius, 24, 24)
sphere_mass = density0 * sphere_radius * sphere_radius * sphere_radius * np.pi * 4 / 3 * 0.7
pile.add(dp.SphereDistance.create(sphere_radius),
         al.uint3(64, 64, 64),
         sign=1,
         thickness=0,
         collision_mesh=sphere_mesh,
         mass=sphere_mass,
         restitution=0,
         friction=0,
         inertia_tensor=dp.f3(1, 1, 1),
         x=unit.from_real_length(dp.f3(0, 0.04, 0)),
         q=dp.f4(0, 0, 0, 1),
         display_mesh=sphere_mesh)

# ## DEBUG
# bunny_mesh = al.Mesh()
# bunny_filename = 'bunny-larger.obj'
# bunny_mesh.set_obj(bunny_filename)
# bunny_mesh.scale(unit.from_real_length(0.1))
# bunny_triangle_mesh = dp.TriangleMesh(bunny_filename)
# bunny_distance = dp.MeshDistance(bunny_triangle_mesh)
# bunny_id = pile.add(bunny_distance,
#                     al.uint3(40, 40, 40),
#                     sign=1,
#                     thickness=0,
#                     collision_mesh=bunny_mesh,
#                     mass=unit.from_real_mass(0.001),
#                     restitution=0.2,
#                     friction=0.8,
#                     inertia_tensor=unit.from_real_moment_of_inertia(
#                         dp.f3(1e-4, 1e-4, 1e-4)),
#                     x=unit.from_real_length(dp.f3(0.07, 0.07, 0.0)),
#                     q=dp.f4(np.sqrt(2), 0, 0, np.sqrt(2)),
#                     display_mesh=bunny_mesh)

pile.build_grids(2 * kernel_radius)
pile.reallocate_kinematics_on_device()
pile.set_gravity(gravity)
cn.contact_tolerance = particle_radius

block_mode = 0
y_shift_due_to_outset = dp.f3(0, real_outset, 0)
box_min = unit.from_real_length(dp.f3(-0.195, 0,
                                      -0.05)) - y_shift_due_to_outset
box_max = unit.from_real_length(dp.f3(-0.095, 0.1,
                                      0.05)) - y_shift_due_to_outset
num_particles = dp.Runner.get_fluid_block_num_particles(
    mode=block_mode,
    box_min=box_min,
    box_max=box_max,
    particle_radius=particle_radius)
container_aabb_range = container_distance.aabb_max - container_distance.aabb_min
container_aabb_range_per_h = container_aabb_range / kernel_radius
grid_res = al.uint3(int(math.ceil(container_aabb_range_per_h.x)),
                    int(math.ceil(container_aabb_range_per_h.y)),
                    int(math.ceil(container_aabb_range_per_h.z))) + 4
grid_offset = al.int3(-(int(grid_res.x) // 2) - 2,
                      -int(math.ceil(real_outset / kernel_radius)) - 1,
                      -(int(grid_res.z) // 2) - 2)

cni.grid_res = grid_res
cni.grid_offset = grid_offset
cni.max_num_particles_per_cell = 64
cni.max_num_neighbors_per_particle = 64

solver = dp.SolverI(runner,
                    pile,
                    dp,
                    num_particles,
                    grid_res,
                    num_ushers=0,
                    enable_surface_tension=False,
                    enable_vorticity=False,
                    graphical=True)
particle_normalized_attr = dp.create_graphical_like(solver.particle_density)
solver.num_particles = num_particles
solver.max_dt = unit.from_real_time(0.1 * unit.rl)
solver.initial_dt = solver.max_dt
solver.min_dt = 0
solver.cfl = 0.4
# solver.density_error_tolerance = 1e-4

dp.copy_cn()

dp.map_graphical_pointers()
runner.launch_create_fluid_block(solver.particle_x,
                                 num_particles,
                                 offset=0,
                                 particle_radius=particle_radius,
                                 mode=block_mode,
                                 box_min=box_min,
                                 box_max=box_max)
dp.unmap_graphical_pointers()

display_proxy.set_camera(unit.from_real_length(al.float3(0, 0.6, 0.6)),
                         unit.from_real_length(al.float3(0, -0.02, 0)))
display_proxy.set_clip_planes(unit.to_real_length(1), box_max.z * 20)
colormap_tex = display_proxy.create_colormap_viridis()

# display_proxy.add_map_graphical_pointers(dp)
# display_proxy.add_step(solver, 10)
# display_proxy.add_normalize(solver, solver.particle_v,
#                             particle_normalized_attr, 0, 2)
# display_proxy.add_normalize(solver, solver.particle_kappa_v, particle_normalized_attr, -0.002, 0)
# display_proxy.add_unmap_graphical_pointers(dp)
display_proxy.add_particle_shading_program(solver.particle_x,
                                           particle_normalized_attr,
                                           colormap_tex,
                                           solver.particle_radius, solver)
display_proxy.add_pile_shading_program(pile)

#display_proxy.run()
for frame_id in range(40000):
    display_proxy.draw()
    dp.map_graphical_pointers()
    for substep_id in range(20):
        solver.step()
        # print(
        #     f"t = {unit.to_real_time(solver.t) } dt = {unit.to_real_time(solver.dt)} cfl = {solver.utilized_cfl} num solves = {solver.num_density_solve}"
        # )
    solver.normalize(solver.particle_v, particle_normalized_attr, 0,
                     unit.from_real_velocity(0.01))
    dp.unmap_graphical_pointers()

dp.remove(particle_normalized_attr)
