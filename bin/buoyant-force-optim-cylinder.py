import alluvion as al
import numpy as np

from util import Unit, parameterize_kinematic_viscosity_with_pellets, BuoySpec

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

unit = Unit(real_kernel_radius=2**-8,
            real_density0=1000,
            real_gravity=-9.80665)

cn.set_kernel_radius(kernel_radius)
cn.set_particle_attr(particle_radius, particle_mass, density0)
cn.gravity = gravity
cn.viscosity, cn.boundary_viscosity = unit.from_real_kinematic_viscosity(
    parameterize_kinematic_viscosity_with_pellets(1e-6))

# rigids
max_num_contacts = 512
capsule_pellet_filename = 'sphere-shell-60mm-2to-8.alu'
capsule_num_pellets = dp.get_alu_info(capsule_pellet_filename)[0][0]
buoy_pellet_filename = 'buoy-2to-8.alu'
buoy_num_pellets = dp.get_alu_info(buoy_pellet_filename)[0][0]
pile = dp.Pile(dp, runner, max_num_contacts, al.VolumeMethod.pellets,
               capsule_num_pellets + buoy_num_pellets)

container_map_radial_size = 32
capsule_pellet_x = dp.create_coated((capsule_num_pellets), 3)
capsule_pellet_x.read_file(capsule_pellet_filename)
sphere_internal_radius = unit.from_real_length(60e-3)
pile.add_pellets(dp.SphereDistance.create(sphere_internal_radius),
                 al.uint3(64, 64, 64),
                 sign=-1,
                 pellets=capsule_pellet_x,
                 mass=0,
                 restitution=0,
                 friction=0,
                 inertia_tensor=dp.f3(1, 1, 1),
                 x=dp.f3(0, 0, 0),
                 q=dp.f4(0, 0, 0, 1),
                 display_mesh=al.Mesh())
dp.remove(capsule_pellet_x)

current_inset = 0.3984

buoy = BuoySpec(dp, unit)
cylinder_neutral_buoyant_force = -buoy.volume * density0 * gravity.y
cylinder_mesh = al.Mesh()
cylinder_mesh.set_cylinder(buoy.radius, buoy.height, 24, 24)
cylinder_mesh.translate(dp.f3(0, -buoy.comy, 0))

buoy_pellet_x = dp.create((buoy_num_pellets), 3)
buoy_pellet_x.read_file(buoy_pellet_filename)
pile.add_pellets(buoy.create_distance(-kernel_radius * 2),
                 buoy.map_dim,
                 sign=1,
                 pellets=buoy_pellet_x,
                 x=dp.f3(0, -sphere_internal_radius * 0.7, 0),
                 mass=0,
                 inertia_tensor=buoy.inertia,
                 display_mesh=cylinder_mesh)
dp.remove(buoy_pellet_x)

pile.reallocate_kinematics_on_device()
pile.set_gravity(gravity)
cn.contact_tolerance = particle_radius

domain_min = pile.domain_min_list[0]
domain_max = pile.domain_max_list[0]
grid_margin = 0
domain_min_multiples = domain_min / kernel_radius
domain_max_multiples = domain_max / kernel_radius
print('domain_min', domain_min)
print('domain_max', domain_max)
grid_min = np.floor([
    domain_min_multiples.x, domain_min_multiples.y, domain_min_multiples.z
]).astype(int) - grid_margin
grid_max = np.ceil([
    domain_max_multiples.x, domain_max_multiples.y, domain_max_multiples.z
]).astype(int) + grid_margin
print('grid_min', grid_min)
print('grid_max', grid_max)
cni.grid_res = al.uint3(*(grid_max - grid_min))
cni.grid_offset = al.int3(*grid_min)
cni.max_num_particles_per_cell = 64
cni.max_num_neighbors_per_particle = 64

## filling
fill_particle_radius = particle_radius * 1.2
fill_mode = 2
fill_margin = 0
fill_domain_min = domain_min - kernel_radius * fill_margin
fill_domain_max = domain_max + kernel_radius * fill_margin

num_sample_positions = dp.Runner.get_fluid_block_num_particles(
    fill_mode, fill_domain_min, fill_domain_max, fill_particle_radius)
print('num_sample_positions', num_sample_positions)
internal_encoded = dp.create((num_sample_positions), 1, np.uint32)
num_particles = pile.compute_sort_fluid_block_internal_all(
    internal_encoded, fill_domain_min, fill_domain_max, fill_particle_radius,
    fill_mode)
print("num_particles", num_particles)
solver = dp.SolverI(runner,
                    pile,
                    dp,
                    num_particles,
                    enable_surface_tension=False,
                    enable_vorticity=False,
                    graphical=True)
particle_normalized_attr = dp.create_graphical((num_particles), 1)

solver.max_dt = unit.from_real_time(0.0001)
solver.initial_dt = solver.max_dt * 0.01
solver.min_dt = 0
solver.cfl = 0.2
solver.num_particles = num_particles
dp.map_graphical_pointers()
runner.launch_create_fluid_block_internal(solver.particle_x, internal_encoded,
                                          num_particles, 0,
                                          fill_particle_radius, fill_mode,
                                          fill_domain_min, fill_domain_max)
dp.unmap_graphical_pointers()
dp.remove(internal_encoded)
display_proxy.set_camera(dp.f3(0, 0, sphere_internal_radius * 4),
                         dp.f3(0, 0, 0))
display_proxy.set_clip_planes(particle_radius * 0.1,
                              sphere_internal_radius * 30)
colormap_tex = display_proxy.create_colormap_viridis()

display_proxy.add_particle_shading_program(solver.particle_x,
                                           particle_normalized_attr,
                                           colormap_tex,
                                           solver.particle_radius * 0.6,
                                           solver)
display_proxy.add_pile_shading_program(pile)

num_stabilized = 0
stabilization_target = 10
buoyant_force_cursor = 0
buoyant_force_buffer = np.zeros(10000, dp.default_dtype)
while True:
    dp.map_graphical_pointers()
    for frame_interstep in range(10):
        solver.step()
        if num_stabilized >= stabilization_target:
            solver.cfl = 0.02
            solver.max_dt = unit.from_real_time(0.00005)
            if buoyant_force_cursor == 0:
                for further_stabilization_step in range(10000):
                    solver.step()
            buoyant_force = runner.sum(solver.particle_force,
                                       solver.particle_force.get_shape()[1],
                                       solver.particle_force.get_shape()[1]).y
            buoyant_force_buffer[buoyant_force_cursor] = buoyant_force
            buoyant_force_cursor += 1
            if (buoyant_force_cursor >= len(buoyant_force_buffer)):
                mean_buoyant_force = np.mean(buoyant_force_buffer)
                print(
                    f'Inset: {current_inset} {mean_buoyant_force}/{cylinder_neutral_buoyant_force}={mean_buoyant_force/cylinder_neutral_buoyant_force}'
                )
                print(buoyant_force_buffer[[1, 5, 500, 600, 2000, 9000, 9999]])
                # reset
                current_inset += unit.from_real_length(5e-7)
                num_stabilized = 0
                stabilization_target = 5
                buoyant_force_cursor = 0

    if num_stabilized >= stabilization_target:
        pass
    else:
        if (int(unit.to_real_time(solver.t) * 1000) % 500 == 0):
            v_rms = np.sqrt(
                runner.sum(solver.particle_cfl_v2, solver.num_particles) /
                solver.num_particles)
            solver.particle_v.set_zero()
            solver.reset_solving_var()
            num_stabilized += 1
            print(
                f"num_stabilized {num_stabilized} vrms={unit.to_real_velocity(v_rms)} max_v={unit.to_real_velocity(np.sqrt(solver.max_v2))}"
            )

    solver.normalize(solver.particle_v, particle_normalized_attr, 0, 2)
    dp.unmap_graphical_pointers()
    display_proxy.draw()
