import math
import time

import alluvion as al
import numpy as np

from util import Unit

dp = al.Depot(np.float32)
cn, cni = dp.create_cn()
dp.create_display(800, 600, "", False)
display_proxy = dp.get_display_proxy()
framebuffer = display_proxy.create_framebuffer()
runner = dp.Runner()
particle_radius = 0.25
kernel_radius = 1.0
density0 = 1.0
cubical_particle_volume = 8 * particle_radius * particle_radius * particle_radius
volume_relative_to_cube = 0.8
particle_mass = cubical_particle_volume * volume_relative_to_cube * density0

gravity = dp.f3(0, -1, 0)

unit = Unit(real_kernel_radius=2**-6,
            real_density0=1000,
            real_gravity=-9.80665)

cn.set_kernel_radius(kernel_radius)
cn.set_particle_attr(particle_radius, particle_mass, density0)
cn.gravity = gravity
cn.viscosity, cn.boundary_viscosity = unit.from_real_kinematic_viscosity(
    np.array([2.049e-6, 6.532e-6]))

cn.inertia_inverse = 0.5
cn.vorticity_coeff = 0.01
cn.viscosity_omega = 0.1

container_pellet_filename = f'cache/cube24-0.015625.alu'

container_num_pellets = dp.get_alu_info(container_pellet_filename)[0][0]
num_pellets = container_num_pellets
# rigids
max_num_contacts = 512
pile = dp.Pile(dp, runner, max_num_contacts, al.VolumeMethod.pellets,
               num_pellets)

## ================== using cube
container_width = unit.from_real_length(0.24)
container_dim = dp.f3(container_width, container_width, container_width)
container_distance = dp.BoxDistance.create(container_dim, outset=0)
container_extent = container_distance.aabb_max - container_distance.aabb_min
container_res_float = container_extent / particle_radius
container_res = al.uint3(int(container_res_float.x),
                         int(container_res_float.y),
                         int(container_res_float.z))
print('container_res', container_res)
container_pellet_x = dp.create((container_num_pellets), 3)
container_pellet_x.read_file(container_pellet_filename)
pile.add_pellets(container_distance,
                 container_res,
                 pellets=container_pellet_x,
                 sign=-1,
                 mass=0,
                 restitution=0.8,
                 friction=0.3)
dp.remove(container_pellet_x)

# ## DEBUG
# bunny_mesh = al.Mesh()
# bunny_filename = 'bunny-larger.obj'
# bunny_mesh.set_obj(bunny_filename)
# bunny_mesh.scale(unit.from_real_length(0.1))
# bunny_triangle_mesh = dp.TriangleMesh(bunny_filename)
# bunny_distance = dp.MeshDistance(bunny_triangle_mesh)
# bunny_id = pile.add(bunny_distance,
#                     collision_mesh=bunny_mesh,
#                     mass=unit.from_real_mass(0.001),
#                     restitution=0.2,
#                     friction=0.8,
#                     x=unit.from_real_length(dp.f3(0.07, 0.07, 0.0)),
#                     q=dp.f4(np.sqrt(2), 0, 0, np.sqrt(2)),
#                     display_mesh=bunny_mesh)

pile.reallocate_kinematics_on_device()
pile.set_gravity(gravity)
cn.contact_tolerance = particle_radius

block_mode = 0
box_min = unit.from_real_length(dp.f3(-0.12, -0.12, -0.12))
box_max = unit.from_real_length(dp.f3(0.12, -0.04, 0.12))
num_particles = dp.Runner.get_fluid_block_num_particles(
    mode=block_mode,
    box_min=box_min,
    box_max=box_max,
    particle_radius=particle_radius)
container_aabb_range_per_h = container_extent / kernel_radius
cni.grid_res = al.uint3(int(math.ceil(container_aabb_range_per_h.x)),
                        int(math.ceil(container_aabb_range_per_h.y)),
                        int(math.ceil(container_aabb_range_per_h.z))) + 4
cni.grid_offset = al.int3(
    int(container_distance.aabb_min.x) - 2,
    int(container_distance.aabb_min.y) - 2,
    int(container_distance.aabb_min.z) - 2)
cni.max_num_particles_per_cell = 64
cni.max_num_neighbors_per_particle = 64

solver = dp.SolverI(runner,
                    pile,
                    dp,
                    num_particles + 10000,
                    num_ushers=0,
                    enable_surface_tension=False,
                    enable_vorticity=False,
                    cn=cn,
                    cni=cni,
                    graphical=True)
particle_normalized_attr = dp.create_graphical_like(solver.particle_density)
solver.num_particles = num_particles
solver.max_dt = unit.from_real_time(0.1 * unit.rl)
solver.initial_dt = solver.max_dt
solver.min_dt = 0
solver.cfl = 0.4
# solver.density_error_tolerance = 1e-4

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
display_proxy.add_bind_framebuffer_step(framebuffer)
display_proxy.add_particle_shading_program(solver.particle_x,
                                           particle_normalized_attr,
                                           colormap_tex,
                                           solver.particle_radius, solver)
display_proxy.add_pile_shading_program(pile)
display_proxy.add_show_framebuffer_shader(framebuffer)

#display_proxy.run()
for frame_id in range(10000):
    display_proxy.draw()
    # framebuffer.write(f"screen{frame_id}.bmp")
    print(framebuffer.width, framebuffer.height)

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
