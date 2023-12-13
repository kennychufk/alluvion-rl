import alluvion as al
import numpy as np
import argparse

from util import Unit

parser = argparse.ArgumentParser(description='Compare with PIV truth')
parser.add_argument('--display', metavar='d', type=bool, default=False)
args = parser.parse_args()


def hcp2(n):
    num_points = n * n * n
    points = np.zeros((num_points, 3))
    for i in range(num_points):
        j = (i % (n * n)) // n
        k = i // (n * n)
        l = i % n
        points[i] = np.array([
            np.sqrt(3) * (j + (k % 2) / 3), 2 * np.sqrt(6) / 3 * k,
            2 * l + (j + k) % 2
        ])
    return points


coord = hcp2(12)

spacing = 0.259967967091019859280982201077
coord *= spacing

dp = al.Depot(np.float64)
cn = dp.cn
cni = dp.cni
if args.display:
    dp.create_display(800, 600, "", False)
display_proxy = dp.get_display_proxy() if args.display else None
runner = dp.Runner()

particle_radius = 0.25
kernel_radius = 1.0
density0 = 1.0
cubical_particle_volume = 8 * particle_radius * particle_radius * particle_radius
volume_relative_to_cube = 0.8
particle_mass = cubical_particle_volume * volume_relative_to_cube * density0

gravity = dp.f3(0, -1, 0)

unit = Unit(real_kernel_radius=1, real_density0=1, real_gravity=-1)

cn.set_kernel_radius(kernel_radius)
cn.set_particle_attr(particle_radius, particle_mass, density0)
cn.gravity = gravity

max_num_contacts = 512
pile = dp.Pile(dp, runner, max_num_contacts)
container_distance = dp.BoxDistance.create(unit.from_real_length(
    dp.f3(26, 26, 26)),
                                           outset=0)
pile.add(container_distance,
         al.uint3(26, 26, 26),
         sign=-1,
         mass=0,
         inertia_tensor=dp.f3(1, 1, 1),
         x=unit.from_real_length(dp.f3(12, 12, 12)))

container_aabb_range = container_distance.aabb_max - container_distance.aabb_min
container_aabb_range_per_h = container_aabb_range / kernel_radius
grid_res = al.uint3(int(np.ceil(container_aabb_range_per_h.x)),
                    int(np.ceil(container_aabb_range_per_h.y)),
                    int(np.ceil(container_aabb_range_per_h.z))) + 4
grid_offset = al.int3(
    -(int(grid_res.x) // 2) - 2,
    -int(np.ceil(container_distance.outset / kernel_radius)) - 1,
    -(int(grid_res.z) // 2) - 2)

cni.grid_res = grid_res
cni.grid_offset = grid_offset
cni.max_num_particles_per_cell = 64
cni.max_num_neighbors_per_particle = 64

use_al_block = False
if use_al_block:
    block_mode = 2
    box_min = dp.f3(0, 0, 0)
    box_max = dp.f3(6, 6, 6)
    num_particles = dp.Runner.get_fluid_block_num_particles(
        mode=block_mode,
        box_min=box_min,
        box_max=box_max,
        particle_radius=spacing)
else:
    num_particles = len(coord)
solver = dp.SolverI(runner,
                    pile,
                    dp,
                    num_particles,
                    num_ushers=0,
                    enable_surface_tension=False,
                    enable_vorticity=False,
                    graphical=args.display)

solver.num_particles = num_particles
solver.max_dt = unit.from_real_time(0.1 * unit.rl)
solver.initial_dt = solver.max_dt
solver.min_dt = 0
solver.cfl = 0.4

dp.map_graphical_pointers()
if use_al_block:
    runner.launch_create_fluid_block(solver.particle_x,
                                     num_particles,
                                     offset=0,
                                     particle_radius=spacing,
                                     mode=block_mode,
                                     box_min=box_min,
                                     box_max=box_max)
else:
    dp.coat(solver.particle_x).set(coord)
dp.unmap_graphical_pointers()

if args.display:
    particle_normalized_attr = dp.create_graphical_like(
        solver.particle_density)
    display_proxy.set_camera(unit.from_real_length(al.float3(40, 40, 40)),
                             unit.from_real_length(al.float3(12, 12, 12)))
    display_proxy.set_clip_planes(unit.to_real_length(1), 12 * 20)
    # display_proxy.set_camera(unit.from_real_length(al.float3(-0.6, 0.6, 0.0)),
    #                          unit.from_real_length(al.float3(0, 0.04, 0)))
    # display_proxy.set_clip_planes(unit.to_real_length(1), 12 * 20)
    colormap_tex = display_proxy.create_colormap_viridis()
    display_proxy.add_particle_shading_program(solver.particle_x,
                                               particle_normalized_attr,
                                               colormap_tex,
                                               solver.particle_radius, solver)

for frame_id in range(1):
    if dp.has_display():
        display_proxy.draw()
    dp.map_graphical_pointers()
    # for substep_id in range(10):
    #     solver.step()
    solver.step()

    densities = dp.coat(solver.particle_density).get()
    print(np.max(densities))
    # xs = dp.coat(solver.particle_x).get()
    # for x in xs:
    #     print(x)
    if dp.has_display():
        solver.normalize(solver.particle_v, particle_normalized_attr, 0,
                         unit.from_real_velocity(0.01))
    dp.unmap_graphical_pointers()

if args.display:
    dp.remove(particle_normalized_attr)
