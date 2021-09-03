import alluvion as al
import sys
import numpy as np

dp = al.Depot(np.float64)
cn = dp.cn
cni = dp.cni
dp.create_display(800, 600, "", False)
display_proxy = dp.get_display_proxy()
runner = dp.Runner()

scale_factor = 2**-11
particle_radius = 0.25 * scale_factor
density0 = 1000
kernel_radius = particle_radius * 4
cubical_particle_volume = 8 * particle_radius * particle_radius * particle_radius
volume_relative_to_cube = 0.8
particle_mass = cubical_particle_volume * volume_relative_to_cube * density0

cn.set_cubic_discretization_constants()
cn.set_kernel_radius(kernel_radius)
cn.set_particle_attr(particle_radius, particle_mass, density0)
cn.boundary_vol_factor = 1.0

current_radius = np.load(sys.argv[2])[0] * scale_factor
pipe_radius_grid_span = int(np.ceil(current_radius / kernel_radius))
pipe_length_grid_half_span = 3
pipe_length_half = pipe_length_grid_half_span * kernel_radius
pipe_length = 2 * pipe_length_grid_half_span * kernel_radius
num_particles = dp.get_alu_info(sys.argv[1])[0][0]
print('num_particles', num_particles)

max_num_contacts = 512
pile = dp.Pile(dp, runner, max_num_contacts)
map_lateral_res = 64
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
solver.particle_x.read_file(sys.argv[1])
solver.particle_x.scale(dp.f3(scale_factor, scale_factor, scale_factor))
dp.unmap_graphical_pointers()

display_proxy.set_camera(
    al.float3(0, pipe_length * 0.2, current_radius * 5.13), al.float3(0, 0, 0))
display_proxy.set_clip_planes(particle_radius * 10, current_radius * 20)
colormap_tex = display_proxy.create_colormap_viridis()
display_proxy.add_particle_shading_program(solver.particle_x,
                                           particle_normalized_attr,
                                           colormap_tex,
                                           solver.particle_radius, solver)

pile.build_grids(kernel_radius)
pile.reallocate_kinematics_on_device()

while True:
    dp.map_graphical_pointers()
    solver.step_wrap1()
    # solver.compute_all_boundaries()
    # solver.update_particle_neighbors_wrap1()
    densities = dp.coat(solver.particle_density).get()
    mean_density = np.mean(densities)
    print('density at', current_radius,
          np.mean(densities) / density0, np.min(densities), np.max(densities))
    solver.normalize(solver.particle_v, particle_normalized_attr, 0, 2)
    dp.unmap_graphical_pointers()
    display_proxy.draw()
