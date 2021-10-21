import alluvion as al
import numpy as np
from pathlib import Path

from util import Unit

dp = al.Depot(np.float32)
cn = dp.cn
cni = dp.cni
dp.create_display(800, 600, "", False)
display_proxy = dp.get_display_proxy()
runner = dp.Runner()

kernel_radius = 1.0
particle_radius = 0.25
density0 = 1.0
cubical_particle_volume = 8 * particle_radius * particle_radius * particle_radius
volume_relative_to_cube = 0.8
particle_mass = cubical_particle_volume * volume_relative_to_cube * density0

gravity = dp.f3(0, -1, 0)

unit = Unit(real_kernel_radius=0.008,
            real_density0=1000,
            real_gravity=-9.80665)

cn.set_kernel_radius(kernel_radius)
cn.set_particle_attr(particle_radius, particle_mass, density0)
cn.boundary_epsilon = 1e-9
cn.gravity = gravity
cn.viscosity, cn.boundary_viscosity = unit.from_real_kinematic_viscosity(
    np.array([2.049e-6, 6.532e-6]))

# rigids
max_num_contacts = 512
pile = dp.Pile(dp, runner, max_num_contacts)

container_width = unit.from_real_length(0.24)
container_dim = dp.f3(container_width, container_width, container_width)
container_mesh = al.Mesh()
container_mesh.set_box(container_dim, 8)
container_distance = dp.BoxDistance(container_dim)
pile.add(container_distance,
         al.uint3(64, 64, 64),
         sign=-1,
         collision_mesh=container_mesh,
         mass=0,
         restitution=0.8,
         friction=0.2,
         inertia_tensor=dp.f3(1, 1, 1),
         x=dp.f3(0, container_width / 2, 0),
         q=dp.f4(0, 0, 0, 1),
         display_mesh=al.Mesh())

pile.reallocate_kinematics_on_device()
pile.set_gravity(gravity)
cn.contact_tolerance = particle_radius

block_mode = 0
edge_factor = 0.49
max_num_particles = int(4232896 / 64)
print('num_particles', max_num_particles)
grid_side = int(np.ceil((container_width + kernel_radius * 2) / kernel_radius))
grid_side += (grid_side % 2 == 1)
# grid_side = 64
grid_res = al.uint3(grid_side, grid_side, grid_side)
print('grid_res', grid_res)
grid_offset = al.int3(-grid_side // 2, -1, -grid_side // 2)

cni.grid_res = grid_res
cni.grid_offset = grid_offset
cni.max_num_particles_per_cell = 64
cni.max_num_neighbors_per_particle = 64

num_ushers = 7
solver = dp.SolverI(runner,
                    pile,
                    dp,
                    max_num_particles,
                    num_ushers=num_ushers,
                    enable_surface_tension=False,
                    enable_vorticity=False,
                    graphical=True)
particle_normalized_attr = dp.create_graphical((max_num_particles), 1)
# print(dp.coat(solver.usher.x).get())
x = np.repeat(0, num_ushers * 3).astype(dp.default_dtype)
v = np.repeat(0, num_ushers * 3).astype(dp.default_dtype)
radius = np.repeat(kernel_radius * 4, num_ushers).astype(dp.default_dtype)
strength = np.repeat(0.01, num_ushers).astype(dp.default_dtype)
solver.usher.set(x, v, radius, strength)

solver.num_particles = max_num_particles
solver.max_dt = unit.from_real_time(0.1 * unit.rl)
solver.initial_dt = solver.max_dt
solver.min_dt = 0
solver.cfl = 0.4

dp.map_graphical_pointers()
runner.launch_create_fluid_cylinder_sunflower(
    solver.particle_x,
    max_num_particles,
    radius=(container_width * 0.5) - kernel_radius,
    num_particles_per_slice=1600,
    slice_distance=particle_radius * 2,
    y_min=kernel_radius)
dp.unmap_graphical_pointers()
display_proxy.set_camera(unit.from_real_length(al.float3(0, 0.06, 0.4)),
                         unit.from_real_length(al.float3(0, 0.06, 0)))
display_proxy.set_clip_planes(unit.to_real_length(1), container_width * 20)
colormap_tex = display_proxy.create_colormap_viridis()

display_proxy.add_particle_shading_program(solver.particle_x,
                                           particle_normalized_attr,
                                           colormap_tex,
                                           solver.particle_radius, solver)
display_proxy.add_pile_shading_program(pile)

next_force_time = 0.0
remaining_force_time = 0.0

fps = 60.0
frame_interval = 1.0 / fps
next_frame_id = 0

while True:
    dp.map_graphical_pointers()
    for frame_interstep in range(20):
        if solver.t >= (next_frame_id * frame_interval):
            next_frame_id += 1
        for usher_id in range(num_ushers):
            ang_f = unit.from_real_angular_velocity(np.pi * 2 * 2)
            phase = np.pi * 2 / num_ushers * usher_id
            theta = ang_f * solver.t + phase
            circle_r = container_width * 0.2
            x[usher_id * 3 + 0] = circle_r * np.sin(theta)
            x[usher_id * 3 +
              1] = circle_r * np.cos(theta) + container_width * 0.3
            v[usher_id * 3 + 0] = circle_r * np.cos(theta) * ang_f
            v[usher_id * 3 + 1] = -circle_r * np.sin(theta) * ang_f
            # strength_temporal =  np.cos(np.pi / 4 * solver.t)
            # strength_temporal = 100
            radius_temporal = np.cos(
                unit.from_real_angular_velocity(np.pi / 2) * solver.t)
            radius_temporal = container_width * 0.01
            print(radius_temporal)
            # strength = unit.from_real_angular_velocity(np.repeat(strength_temporal * strength_temporal,
            #                      num_ushers).astype(dp.default_dtype))
            strength = unit.from_real_angular_velocity(
                np.repeat(100, num_ushers).astype(dp.default_dtype))
            # radius = np.repeat(
            #     kernel_radius * (4 + 4 * radius_temporal * radius_temporal),
            #     num_ushers).astype(dp.default_dtype)
            radius = np.repeat(
                kernel_radius * (4 + 4 * radius_temporal * radius_temporal),
                num_ushers).astype(dp.default_dtype)
            solver.usher.set(x, v, radius, strength)
        solver.step()

    solver.normalize(solver.particle_v, particle_normalized_attr, 0, 2)
    dp.unmap_graphical_pointers()
    display_proxy.draw()
