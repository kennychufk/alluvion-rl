import alluvion as al
import sys
import numpy as np
from numpy import linalg as LA
import time

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

cn.set_kernel_radius(kernel_radius)
cn.set_particle_attr(particle_radius, particle_mass, density0)
cn.viscosity = 0
cn.boundary_viscosity = 0
cn.contact_tolerance = 0

pipe_radius_grid_span = int(sys.argv[1])
current_radius = kernel_radius * pipe_radius_grid_span
pipe_length_grid_half_span = 3
pipe_length_half = pipe_length_grid_half_span * kernel_radius
pipe_length = 2 * pipe_length_grid_half_span * kernel_radius
num_particles = dp.Runner.get_fluid_cylinder_num_particles(
    current_radius, -pipe_length_half, pipe_length_half, particle_radius)
print('num_particles', num_particles)

max_num_contacts = 0
# pile for pellet filling
max_num_pellets = 50000
fill_pile = dp.Pile(dp,
                    runner,
                    max_num_contacts=0,
                    volume_method=al.VolumeMethod.pellets,
                    max_num_pellets=0)
fill_pile.add(dp.InfiniteTubeDistance.create(current_radius,
                                             current_radius + kernel_radius,
                                             pipe_length_half),
              al.uint3(30, 1, 30),
              sign=-1)
fill_pile.reallocate_kinematics_on_device()
pellet_normalized_attr = dp.create_graphical((max_num_pellets), 1)

pile = dp.Pile(dp, runner, max_num_contacts, al.VolumeMethod.pellets,
               max_num_pellets)
map_lateral_res = 32
pile.add(dp.InfiniteCylinderDistance.create(current_radius),
         al.uint3(map_lateral_res, 1, map_lateral_res),
         sign=-1)
pile.reallocate_kinematics_on_device()

grid_radial_margin_for_pellets = 4
cni.grid_res = al.uint3(
    (pipe_radius_grid_span + grid_radial_margin_for_pellets) * 2,
    pipe_length_grid_half_span * 2,
    (pipe_radius_grid_span + grid_radial_margin_for_pellets) * 2)
cni.grid_offset = al.int3(
    -pipe_radius_grid_span - grid_radial_margin_for_pellets,
    -pipe_length_grid_half_span,
    -pipe_radius_grid_span - grid_radial_margin_for_pellets)
cn.set_wrap_length(cni.grid_res.y * kernel_radius)
cni.max_num_particles_per_cell = 64
cni.max_num_neighbors_per_particle = 64

solver = dp.SolverI(runner,
                    pile,
                    dp,
                    num_particles,
                    enable_surface_tension=False,
                    enable_vorticity=False,
                    graphical=True)
particle_normalized_attr = dp.create_graphical((num_particles), 1)
solver.num_particles = num_particles
solver.dt = 1e-3
solver.max_dt = 1e-3
solver.min_dt = 0.0
solver.cfl = 0.05

fill_solver = dp.SolverPellet(runner,
                              fill_pile,
                              dp,
                              max_num_pellets,
                              graphical=True)
print('fill_solver shapes', fill_solver.pid.get_shape(),
      fill_solver.pid_length.get_shape())

dp.map_graphical_pointers()
solver.particle_x.read_file('.alcache/9900.alu')
previous_bead_x = dp.coat(solver.particle_x).get(num_particles)
dp.unmap_graphical_pointers()

display_proxy.set_camera(
    al.float3(0, pipe_length * 0.2, current_radius * 5.13), al.float3(0, 0, 0))
colormap_tex = display_proxy.create_colormap_viridis()
display_proxy.add_particle_shading_program(solver.particle_x,
                                           particle_normalized_attr,
                                           colormap_tex,
                                           solver.particle_radius,
                                           solver,
                                           clear=True)
display_proxy.add_particle_shading_program(fill_solver.particle_x,
                                           pellet_normalized_attr,
                                           colormap_tex,
                                           fill_solver.particle_radius,
                                           fill_solver,
                                           clear=False)

radius_decrease_rate = 2**-7
best_density = 0.0
best_radius = 0.0
best_density_diff = 1.0
best_x = dp.create((num_particles), 3)

stage = 0
current_radius = 7.375
max_pellet_density = 0
pellet_initialized = False
num_steps = 200
num_fill_steps = 1000
while True:
    current_radius -= radius_decrease_rate
    outer_radius = current_radius + kernel_radius * 3

    # ===== remember old settings
    old_density0 = cn.density0
    old_particle_mass = cn.particle_mass
    # old_max_num_contacts = cni.max_num_contacts
    old_num_boundaries = cni.num_boundaries
    # ===== overwrite with fill settings
    fill_density0 = 1000
    fill_particle_mass = kernel_radius * kernel_radius * kernel_radius * 0.125 * density0
    cn.set_particle_attr(cn.particle_mass, fill_particle_mass, fill_density0)
    # ===== create regular pellets
    fill_pile.replace(0,
                      dp.InfiniteTubeDistance.create(current_radius,
                                                     outer_radius),
                      al.uint3(30, 1, 30),
                      sign=-1)
    fill_pile.reallocate_kinematics_on_device()
    domain_min = fill_pile.domain_min_list[0]
    domain_max = fill_pile.domain_max_list[0]
    fill_mode = 2
    fill_particle_radius = particle_radius * 0.59
    if not pellet_initialized:
        num_sample_positions = dp.Runner.get_fluid_cylinder_num_particles(
            outer_radius, -pipe_length_half, pipe_length_half,
            fill_particle_radius)
        internal_encoded = dp.create((num_sample_positions), 1, np.uint32)
        num_pellets = fill_pile.compute_sort_fluid_cylinder_internal_all(
            internal_encoded, outer_radius, fill_particle_radius,
            -pipe_length_half, pipe_length_half)
        print("num_pellets for filling", num_pellets)
        x = dp.create((num_pellets), 3)
        runner.launch_create_fluid_cylinder_internal(
            x, internal_encoded, num_pellets, 0, outer_radius,
            fill_particle_radius, -pipe_length_half, pipe_length_half)
        dp.map_graphical_pointers()
        fill_solver.set_pellets(x)
        dp.unmap_graphical_pointers()
        dp.remove(internal_encoded)
        pellet_initialized = True
    # pellet_v = dp.coat(fill_solver.particle_v).get()[:num_pellets]
    # print('pellet_v.shape', pellet_v.shape)
    # pellet_v_norm = LA.norm(pellet_v, axis=1)
    # pellet_v_mean = np.mean(pellet_v_norm)
    # print('pellet_v max', np.max(pellet_v_norm), 'mean', pellet_v_mean, 'min', np.min(pellet_v_norm))
    # if pellet_v_mean > 0.03:
    #     num_pellets -= int((pellet_v_mean - 0.03)* 10000)
    #     fill_solver.num_particles = num_pellets
    pellet_density = dp.coat(fill_solver.particle_density).get()[:num_pellets]
    max_pellet_density = np.max(pellet_density)
    min_pellet_density = np.min(pellet_density)
    mean_pellet_density = np.mean(pellet_density)
    print('pellet_density', min_pellet_density, mean_pellet_density,
          max_pellet_density)
    if min_pellet_density > 0.98:
        num_pellets -= int((min_pellet_density - 0.98) * 100)
        fill_solver.num_particles = num_pellets

    num_fill_substeps = 100
    num_major_fill_steps = num_fill_steps // num_fill_substeps
    for major_fill_step_id in range(num_major_fill_steps):
        display_proxy.draw()
        dp.map_graphical_pointers()
        for substep_id in range(num_fill_substeps):
            # print('solve fill')
            # print('fill solver num_particles', fill_solver.num_particles, 'max_num_particles', fill_solver.max_num_particles)
            # print('fill pile num_pellets', fill_pile.num_pellets)
            fill_solver.step_wrap1()
        fill_solver.normalize(fill_solver.particle_v, pellet_normalized_attr,
                              0, kernel_radius * 0.2)
        dp.unmap_graphical_pointers()
    print('finished initializing pellets')

    # ===== restore old settings
    cn.set_particle_attr(cn.particle_radius, old_particle_mass, old_density0)
    # cni.max_num_contacts = old_max_num_contacts
    cni.num_boundaries = old_num_boundaries

    dp.map_graphical_pointers()
    pellet_x = dp.create((num_pellets), 3)
    pellet_x.set_from(fill_solver.particle_x, num_pellets)
    pile.replace_pellets(0,
                         dp.InfiniteCylinderDistance.create(current_radius),
                         al.uint3(map_lateral_res, 1, map_lateral_res),
                         pellets=pellet_x,
                         sign=-1)
    dp.remove(pellet_x)
    dp.unmap_graphical_pointers()
    pile.reallocate_kinematics_on_device()
    print("pile.num_pellets", pile.num_pellets)
    print("solver.num_particles", solver.num_particles)

    num_substeps = 50
    num_major_steps = num_steps // num_substeps
    for major_step_id in range(num_major_steps):
        display_proxy.draw()
        dp.map_graphical_pointers()
        solver.particle_v.set_zero()
        solver.reset_solving_var()
        for substep_id in range(num_substeps):
            solver.step_wrap1()
        solver.normalize(solver.particle_density, particle_normalized_attr,
                         density0 * 0.5, density0 * 1.2)
        dp.unmap_graphical_pointers()
    dp.map_graphical_pointers()
    bead_x = dp.coat(solver.particle_x).get(num_particles)
    print('abs diff bead x', np.sum(np.abs(previous_bead_x - bead_x)))
    previous_bead_x = bead_x
    densities = dp.coat(solver.particle_density).get(num_particles)
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
        radius_decrease_rate *= 0.5
        print("===========", radius_decrease_rate)
    elif stage == 4 and (density0 - mean_density) < 2**-12:
        num_steps = 1000
        stage += 1
        radius_decrease_rate *= 0.0625
        solver.cfl = 0.015625
        print("===========", radius_decrease_rate)
    elif stage == 5 and (density0 - mean_density) < 2**-13:
        num_steps = 3000
        num_fill_steps = 2000
        stage += 1
        radius_decrease_rate *= 0.25
        print("===========", radius_decrease_rate)
    elif stage == 6 and (density0 - mean_density) < 2**-14:
        num_steps = 5000
        num_fill_steps = 3000
        stage += 1
        radius_decrease_rate *= 0.5
        print("===========", radius_decrease_rate)
    if (mean_density <= 1 and best_density <= mean_density):
        density_diff = np.max(densities) - np.min(densities)
        if best_density < mean_density or density_diff < best_density_diff:
            best_density = mean_density
            best_density_diff = density_diff
            best_x.set_from(solver.particle_x, num_particles)
            best_radius = current_radius
            print("!! Best", best_radius, best_density, best_density_diff)
            np.save(f".alcache/shrink-best-stat{solver.num_particles}",
                    np.array([best_radius, best_density, best_density_diff]))
            best_x.write_file(
                f".alcache/shrink-best-{solver.num_particles}.alu",
                solver.num_particles)
            fill_solver.particle_x.write_file(
                f".alcache/shrink-best-{solver.num_particles}-pellets.alu",
                num_pellets)
    best_x.write_file(f".alcache/shrink-latest-{solver.num_particles}.alu",
                      num_particles)
    fill_solver.particle_x.write_file(
        f".alcache/shrink-latest-{solver.num_particles}-pellets.alu",
        num_pellets)
    dp.unmap_graphical_pointers()
    if (mean_density > 1):
        break
    print("================", stage)

dp.remove(particle_normalized_attr)
dp.remove(pellet_normalized_attr)
