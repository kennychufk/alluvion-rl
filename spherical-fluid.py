import alluvion as al
import sys
import numpy as np

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

cn.set_cubic_discretization_constants()
cn.set_kernel_radius(kernel_radius)
cn.set_particle_attr(particle_radius, particle_mass, density0)
cn.boundary_vol_factor = 1.0

num_particles = 6 * 6 * 6
side_length = pow(num_particles, 1 / 3)
half_box_extent = (side_length + 1) * particle_radius
print(half_box_extent)

max_num_contacts = 512
pile = dp.Pile(dp, runner, max_num_contacts)
current_radius = side_length / 2
pile.add(dp.SphereDistance.create(current_radius),
         al.uint3(32, 32, 32),
         sign=-1)
pile.build_grids(kernel_radius)
pile.reallocate_kinematics_on_device()

grid_half_extent = int(half_box_extent / kernel_radius) + 1
print("grid_half_extent", grid_half_extent)
cni.grid_res = al.uint3(grid_half_extent * 2 + 1, grid_half_extent * 2 + 1,
                        grid_half_extent * 2 + 1)
cni.grid_offset = al.int3(-grid_half_extent, -grid_half_extent,
                          -grid_half_extent)
cni.max_num_particles_per_cell = 64
cni.max_num_neighbors_per_particle = 64
solver = dp.SolverDf(runner,
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
                                 mode=0,
                                 particle_radius=particle_radius,
                                 box_min=dp.f3(-half_box_extent,
                                               -half_box_extent,
                                               -half_box_extent),
                                 box_max=dp.f3(half_box_extent,
                                               half_box_extent,
                                               half_box_extent))
dp.unmap_graphical_pointers()

display_proxy.set_camera(al.float3(0, 6, 6), al.float3(0, 0, 0))
colormap_tex = display_proxy.create_colormap_viridis()
display_proxy.add_particle_shading_program(solver.particle_x,
                                           particle_normalized_attr,
                                           colormap_tex,
                                           solver.particle_radius, solver)

radius_decrease_rate = 2**-7
map_lateral_res = 16
stage = 0
for frame_id in range(1000):
    # if frame_id < 40:
    #     current_radius -= 0.02
    # elif frame_id < 40 + 8:
    #     current_radius -= 8e-3
    # else:
    #     current_radius -= 1e-5

    current_radius -= radius_decrease_rate
    pile.replace(0,
                 dp.SphereDistance.create(current_radius),
                 al.uint3(map_lateral_res, map_lateral_res, map_lateral_res),
                 sign=-1)
    pile.build_grids(kernel_radius)
    pile.reallocate_kinematics_on_device()
    dp.map_graphical_pointers()
    for substep_id in range(1000):
        solver.step()
        if substep_id % 50 == 0:
            solver.particle_v.set_zero()
            solver.reset_solving_var()
    densities = dp.coat(solver.particle_density).get()
    mean_density = np.mean(densities)
    print('density at', current_radius, mean_density, densities[:100])
    if stage == 0 and (density0 - mean_density) < 1e-2:
        radius_decrease_rate *= 0.25
        stage += 1
        print("===========", radius_decrease_rate)
    elif stage == 1 and (density0 - mean_density) < 1e-2 * 0.5:
        stage += 1
        radius_decrease_rate *= 0.25
        print("===========", radius_decrease_rate)
    elif stage == 2 and (density0 - mean_density) < 1e-2 * 0.25:
        stage += 1
        radius_decrease_rate *= 0.25
        print("===========", radius_decrease_rate)
    elif stage == 3 and (density0 - mean_density) < 1e-2 * 0.125:
        stage += 1
        # radius_decrease_rate *= 0.25
        radius_decrease_rate = 1e-5
        map_lateral_res = 64
        solver.cfl = 1e-2
        print("===========", radius_decrease_rate)
    if (mean_density > 1):
        sys.exit()
    solver.normalize(solver.particle_v, particle_normalized_attr, 0, 2)
    dp.unmap_graphical_pointers()
    display_proxy.draw()
