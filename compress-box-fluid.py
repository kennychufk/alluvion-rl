import alluvion as al
import sys
import numpy as np
import math

from util import Unit

dp = al.Depot(np.float32)
cn = dp.cn
cni = dp.cni
dp.create_display(800, 600, "", False)
display_proxy = dp.get_display_proxy()
runner = dp.Runner()

particle_radius = 0.25
density0 = 1
kernel_radius = particle_radius * 4
cubical_particle_volume = 8 * particle_radius * particle_radius * particle_radius
volume_relative_to_cube = 0.8
particle_mass = cubical_particle_volume * volume_relative_to_cube * density0

unit = Unit(real_kernel_radius=0.008,
            real_density0=998.91,
            real_gravity=-9.80665)

cn.set_cubic_discretization_constants()
cn.set_kernel_radius(kernel_radius)
cn.set_particle_attr(particle_radius, particle_mass, density0)

real_base_width = 0.24
real_fluid_height0 = 0.06
fluid_block_mode = 0
fluid_block_min = unit.from_real_length(
    dp.f3(real_base_width * -0.5, 0, real_base_width * -0.5))
fluid_block_max = unit.from_real_length(
    dp.f3(real_base_width * 0.5, real_fluid_height0, real_base_width * 0.5))
num_particles = dp.Runner.get_fluid_block_num_particles(
    mode=fluid_block_mode,
    box_min=fluid_block_min,
    box_max=fluid_block_max,
    particle_radius=particle_radius)
real_fluid_height = unit.to_real_mass(
    particle_mass
) * num_particles / unit.rdensity0 / real_base_width / real_base_width
print('real_fluid_height', real_fluid_height)

print('num_particles', num_particles)
current_outset = kernel_radius * 2

max_num_contacts = 512
pile = dp.Pile(dp, runner, max_num_contacts)
map_base_res = 32
map_height_res = int(32 / real_base_width * real_fluid_height)
container_distance = dp.BoxDistance.create(unit.from_real_length(
    dp.f3(real_base_width, real_fluid_height, real_base_width)),
                                           outset=current_outset)
pile.add(container_distance,
         al.uint3(map_base_res, map_height_res, map_base_res),
         sign=-1,
         x=unit.from_real_length(dp.f3(0, real_fluid_height * 0.5, 0)))
pile.build_grids(kernel_radius)
pile.reallocate_kinematics_on_device()

container_aabb_range = container_distance.aabb_max - container_distance.aabb_min
container_aabb_range_per_h = container_aabb_range / kernel_radius
cni.grid_res = al.uint3(int(math.ceil(container_aabb_range_per_h.x)),
                        int(math.ceil(container_aabb_range_per_h.y)),
                        int(math.ceil(container_aabb_range_per_h.z))) + 4
cni.grid_offset = al.int3(-(int(cni.grid_res.x) // 2) - 2,
                          -int(math.ceil(current_outset / kernel_radius)) - 1,
                          -(int(cni.grid_res.z) // 2) - 2)
cni.max_num_particles_per_cell = 64
cni.max_num_neighbors_per_particle = 64
# cn.gravity = dp.f3(0, 90.1, 0)

solver = dp.SolverI(runner,
                    pile,
                    dp,
                    num_particles,
                    cni.grid_res,
                    enable_surface_tension=False,
                    enable_vorticity=False,
                    graphical=True)
particle_normalized_attr = dp.create_graphical((num_particles), 1)
solver.num_particles = num_particles
solver.dt = 1e-3
solver.max_dt = 1e-3
solver.min_dt = 0.0
solver.cfl = 2e-2
dp.copy_cn()

dp.map_graphical_pointers()
runner.launch_create_fluid_block(solver.particle_x,
                                 num_particles,
                                 offset=0,
                                 particle_radius=particle_radius,
                                 mode=fluid_block_mode,
                                 box_min=fluid_block_min,
                                 box_max=fluid_block_max)
dp.unmap_graphical_pointers()

display_proxy.set_camera(
    unit.from_real_length(
        al.float3(0, real_fluid_height * 0.2, real_base_width * 2)),
    al.float3(0, 0, 0))
colormap_tex = display_proxy.create_colormap_viridis()
display_proxy.add_particle_shading_program(solver.particle_x,
                                           particle_normalized_attr,
                                           colormap_tex,
                                           solver.particle_radius, solver)

outset_decrease_rate = 2**-6
best_density = 0.0
best_outset = 0.0
best_density_diff = 1.0
best_x = dp.create((num_particles), 3)

stage = 0
while True:
    current_outset -= outset_decrease_rate
    container_distance_new = dp.BoxDistance.create(unit.from_real_length(
        dp.f3(real_base_width, real_fluid_height, real_base_width)),
                                                   outset=current_outset)
    pile.replace(0,
                 container_distance_new,
                 al.uint3(map_base_res, map_height_res, map_base_res),
                 sign=-1,
                 x=unit.from_real_length(dp.f3(0, real_fluid_height * 0.5, 0)))
    pile.build_grids(kernel_radius)
    pile.reallocate_kinematics_on_device()

    dp.map_graphical_pointers()
    for substep_id in range(1000):
        solver.step()
        if (substep_id % 50 == 0):
            solver.particle_v.set_zero()
            solver.reset_solving_var()
    densities = dp.coat(solver.particle_density).get()
    mean_density = np.mean(densities)
    print('density at', current_outset, np.mean(densities), np.min(densities),
          np.max(densities))
    if stage == 0 and (density0 - mean_density) < 1e-2:
        outset_decrease_rate *= 0.25
        stage += 1
        print("===========", outset_decrease_rate)
    elif stage == 1 and (density0 - mean_density) < 1e-2 * 0.5:
        stage += 1
        outset_decrease_rate *= 0.5
        print("===========", outset_decrease_rate)
    elif stage == 2 and (density0 - mean_density) < 1e-2 * 0.25:
        stage += 1
        outset_decrease_rate *= 0.5
        print("===========", outset_decrease_rate)
    elif stage == 3 and (density0 - mean_density) < 1e-2 * 0.125:
        stage += 1
        # outset_decrease_rate *= 0.25
        outset_decrease_rate = 1e-5
        map_lateral_res = 64
        solver.cfl = 1e-2
        print("===========", outset_decrease_rate)
    if (mean_density <= 1 and best_density <= mean_density):
        density_diff = np.max(densities) - np.min(densities)
        if best_density < mean_density or density_diff < best_density_diff:
            best_density = mean_density
            best_density_diff = density_diff
            best_x.set_from(solver.particle_x)
            best_outset = current_outset
            print("!! Best", best_outset, best_density, best_density_diff)
    if (mean_density > 1):
        break
    solver.normalize(solver.particle_v, particle_normalized_attr, 0, 2)
    dp.unmap_graphical_pointers()
    display_proxy.draw()

best_x.write_file(f".alcache/{solver.num_particles}.alu", solver.num_particles)
np.save(f".alcache/stat{solver.num_particles}",
        np.array([best_outset, best_density, best_density_diff]))
