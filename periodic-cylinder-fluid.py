import alluvion as al
import sys
import numpy as np

dp = al.Depot(np.float64)
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

pipe_radius_grid_span = int(sys.argv[1])
current_radius = kernel_radius * pipe_radius_grid_span
pipe_length_grid_half_span = 3
pipe_length_half = pipe_length_grid_half_span * kernel_radius
pipe_length = 2 * pipe_length_grid_half_span * kernel_radius
num_particles = dp.Runner.get_fluid_cylinder_num_particles(
    current_radius, -pipe_length_half, pipe_length_half, particle_radius)
print('num_particles', num_particles)

max_num_contacts = 512
pile = dp.Pile(dp, runner, max_num_contacts)
map_lateral_res = 32
pile.add(dp.InfiniteCylinderDistance.create(current_radius),
         al.uint3(map_lateral_res, 1, map_lateral_res),
         sign=-1)
pile.build_grids(kernel_radius)
pile.reallocate_kinematics_on_device()

cni.grid_res = al.uint3(pipe_radius_grid_span * 2,
                        pipe_length_grid_half_span * 2,
                        pipe_radius_grid_span * 2)
cni.grid_offset = al.int3(-pipe_radius_grid_span, -pipe_length_grid_half_span,
                          -pipe_radius_grid_span)
cni.max_num_particles_per_cell = 64
cni.max_num_neighbors_per_particle = 64
# cn.gravity = dp.f3(0, 90.1, 0)
cn.set_wrap_length(cni.grid_res.y * kernel_radius)

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
solver.particle_radius = particle_radius
dp.copy_cn()

dp.map_graphical_pointers()
runner.launch_create_fluid_cylinder(solver.particle_x,
                                    num_particles,
                                    offset=0,
                                    radius=current_radius,
                                    y_min=-pipe_length_half,
                                    y_max=pipe_length_half)
dp.unmap_graphical_pointers()

display_proxy.set_camera(
    al.float3(0, pipe_length * 0.2, current_radius * 5.13), al.float3(0, 0, 0))
colormap_tex = display_proxy.create_colormap_viridis()
display_proxy.add_particle_shading_program(solver.particle_x,
                                           particle_normalized_attr,
                                           colormap_tex,
                                           solver.particle_radius, solver)

radius_decrease_rate = 2**-7
best_density = 0.0
best_radius = 0.0
best_density_diff = 1.0
best_x = dp.create((num_particles), 3)

stage = 0
while True:
    current_radius -= radius_decrease_rate
    pile.replace(0,
                 dp.InfiniteCylinderDistance.create(current_radius),
                 al.uint3(map_lateral_res, map_lateral_res, map_lateral_res),
                 sign=-1)
    pile.build_grids(kernel_radius)
    pile.reallocate_kinematics_on_device()

    dp.map_graphical_pointers()
    for substep_id in range(1000):
        solver.step_wrap1()
        if (substep_id % 50 == 0):
            solver.particle_v.set_zero()
            solver.reset_solving_var()
    densities = dp.coat(solver.particle_density).get()
    mean_density = np.mean(densities)
    print('density at', current_radius, np.mean(densities), np.min(densities),
          np.max(densities))
    if stage == 0 and (density0 - mean_density) < 1e-2:
        radius_decrease_rate *= 0.25
        stage += 1
        print("===========", radius_decrease_rate)
    elif stage == 1 and (density0 - mean_density) < 1e-2 * 0.5:
        stage += 1
        radius_decrease_rate *= 0.5
        print("===========", radius_decrease_rate)
    elif stage == 2 and (density0 - mean_density) < 1e-2 * 0.25:
        stage += 1
        radius_decrease_rate *= 0.5
        print("===========", radius_decrease_rate)
    elif stage == 3 and (density0 - mean_density) < 1e-2 * 0.125:
        stage += 1
        # radius_decrease_rate *= 0.25
        radius_decrease_rate = 1e-5
        map_lateral_res = 64
        solver.cfl = 1e-2
        print("===========", radius_decrease_rate)
    if (mean_density <= 1 and best_density <= mean_density):
        density_diff = np.max(densities) - np.min(densities)
        if best_density < mean_density or density_diff < best_density_diff:
            best_density = mean_density
            best_density_diff = density_diff
            best_x.set_from(solver.particle_x)
            best_radius = current_radius
            print("!! Best", best_radius, best_density, best_density_diff)
    if (mean_density > 1):
        break
    solver.normalize(solver.particle_v, particle_normalized_attr, 0, 2)
    dp.unmap_graphical_pointers()
    display_proxy.draw()

best_x.write_file(f".alcache/{solver.num_particles}.alu", solver.num_particles)
np.save(f".alcache/stat{solver.num_particles}",
        np.array([best_radius, best_density, best_density_diff]))
