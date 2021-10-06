import alluvion as al
import numpy as np

dp = al.Depot(np.float64)
cn = dp.cn
cni = dp.cni
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

max_num_contacts = 512
pile = dp.Pile(dp, runner, max_num_contacts)
pile.add(dp.SphereDistance.create(0.667115443061774),
         al.uint3(32, 32, 32),
         sign=-1,
         thickness=0,
         collision_mesh=al.Mesh(),
         mass=0,
         restitution=1,
         friction=0,
         inertia_tensor=dp.f3(1, 1, 1),
         x=dp.f3(0, 0, 0),
         q=dp.f4(0, 0, 0, 1),
         display_mesh=al.Mesh())
pile.build_grids(kernel_radius)
pile.reallocate_kinematics_on_device()

num_particles = 1
cni.grid_res = al.uint3(5, 5, 5)
cni.grid_offset = al.int3(-2, -2, -2)
cni.max_num_particles_per_cell = 64
cni.max_num_neighbors_per_particle = 64
solver = dp.SolverDf(runner,
                     pile,
                     dp,
                     num_particles,
                     cni.grid_res,
                     enable_surface_tension=False,
                     enable_vorticity=False,
                     graphical=False)
solver.num_particles = num_particles
solver.dt = 1e-3
solver.max_dt = 1e-3
solver.min_dt = 0.0
solver.cfl = 2e-2
dp.coat(solver.particle_x).set(np.zeros(3))
dp.copy_cn()

print('x before', dp.coat(solver.particle_x).get())
solver.step()
print('x after', dp.coat(solver.particle_x).get())
print('boundary', dp.coat(solver.particle_boundary).get())
print('boundary_kernel', dp.coat(solver.particle_boundary_kernel).get())
print('density', dp.coat(solver.particle_density).get())
