import alluvion as al
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='RL ground truth generator')
parser.add_argument('--input', type=str, default='')
args = parser.parse_args()
dp = al.Depot(np.float32)
cn = dp.cn
cni = dp.cni
dp.create_display(800, 600, "", False)
display_proxy = dp.get_display_proxy()
runner = dp.Runner()

particle_radius = 2**-11
kernel_radius = particle_radius * 4
density0 = 1000.0
cubical_particle_volume = 8 * particle_radius * particle_radius * particle_radius
volume_relative_to_cube = 0.8
particle_mass = cubical_particle_volume * volume_relative_to_cube * density0
gravity = dp.f3(0, -9.81, 0)

cn.set_cubic_discretization_constants()
cn.set_kernel_radius(kernel_radius)
cn.set_particle_attr(particle_radius, particle_mass, density0)
cn.boundary_epsilon = 1e-9
cn.gravity = gravity
cn.viscosity = 1.37916076e-05
cn.boundary_viscosity = 4.96210578e-07

# rigids
max_num_contacts = 512
pile = dp.Pile(dp, max_num_contacts)

container_width = 0.24
container_dim = dp.f3(container_width, container_width, container_width)
container_mesh = al.Mesh()
container_mesh.set_box(container_dim, 8)
container_distance = dp.BoxDistance(container_dim)
pile.add(container_distance,
         al.uint3(64, 64, 64),
         sign=-1,
         thickness=kernel_radius,
         collision_mesh=container_mesh,
         mass=0,
         restitution=0,
         friction=0,
         inertia_tensor=dp.f3(1, 1, 1),
         x=dp.f3(0, container_width / 2, 0),
         q=dp.f4(0, 0, 0, 1),
         display_mesh=al.Mesh())

# bunny_mesh = al.Mesh()
# bunny_filename = 'bunny-foo.obj'
# bunny_mesh.set_obj(bunny_filename)
# bunny_triangle_mesh = dp.TriangleMesh(bunny_filename)
# bunny_distance = dp.MeshDistance(bunny_triangle_mesh)
# pile.add(bunny_distance,
#         al.uint3(40, 40, 40),
#         sign=1,
#         thickness = 0,
#         collision_mesh = al.Mesh(),
#         mass=0,
#         restitution=0.8,
#         friction =0.3,
#         inertia_tensor = dp.f3(1, 1, 1),
#         x=dp.f3(0, container_width * 0.7, 0),
#         q=dp.f4(0, 0, 0, 1),
#         display_mesh = bunny_mesh)

inset_factor = 1.71

cylinder_radius = 3.00885e-3
cylinder_height = 38.5e-3
cylinder_comy = -8.8521e-3
cylinder_mass = 1.06e-3
cylinder_mesh = al.Mesh()
cylinder_mesh.set_cylinder(cylinder_radius, cylinder_height, 24, 24)
cylinder_mesh.translate(dp.f3(0, -cylinder_comy, 0))
cylinder_distance = dp.CylinderDistance(cylinder_radius, cylinder_height,
                                        cylinder_comy)

num_buoys = 7
for i in range(num_buoys):
    pile.add(cylinder_distance,
             al.uint3(20, 128, 20),
             sign=1,
             thickness=-particle_radius * inset_factor,
             collision_mesh=cylinder_mesh,
             mass=cylinder_mass,
             restitution=0,
             friction=0,
             inertia_tensor=dp.f3(7.91134e-8, 2.94462e-9, 7.91134e-8),
             x=dp.f3(
                 np.random.uniform(-container_width * 0.45,
                                   container_width * 0.45),
                 container_width * np.random.uniform(0.55, 0.75),
                 np.random.uniform(-container_width * 0.45,
                                   container_width * 0.45)),
             q=dp.f4(0, 0, 0, 1),
             display_mesh=cylinder_mesh)

pile.build_grids(4 * kernel_radius)
pile.reallocate_kinematics_on_device()
pile.set_gravity(gravity)
cn.contact_tolerance = particle_radius

block_mode = 0
edge_factor = 0.49
box_min = dp.f3((container_width - 2 * kernel_radius) * -edge_factor,
                kernel_radius,
                (container_width - kernel_radius * 2) * -edge_factor)
box_max = dp.f3((container_width - 2 * kernel_radius) * edge_factor,
                container_width * 0.32,
                (container_width - kernel_radius * 2) * edge_factor)
# box_min = dp.f3(container_width * -0.45, 0,  container_width * -0.45)
# box_max = dp.f3(0, container_width * 0.46, 0)
max_num_particles = dp.Runner.get_fluid_block_num_particles(
    mode=block_mode,
    box_min=box_min,
    box_max=box_max,
    particle_radius=particle_radius)
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

solver = dp.SolverDf(runner,
                     pile,
                     dp,
                     max_num_particles,
                     grid_res,
                     enable_surface_tension=False,
                     enable_vorticity=False,
                     graphical=True)
particle_normalized_attr = dp.create_graphical((max_num_particles), 1)

solver.dt = 1e-3
solver.max_dt = 5e-5
solver.min_dt = 0.0
solver.cfl = 0.04
solver.particle_radius = particle_radius
solver.num_particles = max_num_particles

dp.copy_cn()

dp.map_graphical_pointers()
if len(args.input) == 0:
    runner.launch_create_fluid_block(256,
                                     solver.particle_x,
                                     solver.num_particles,
                                     offset=0,
                                     mode=block_mode,
                                     box_min=box_min,
                                     box_max=box_max)
else:
    solver.particle_x.read_file(f'{args.input}-x.alu')
    solver.particle_v.read_file(f'{args.input}-v.alu')
    pile.read_file(f'{args.input}.pile')
dp.unmap_graphical_pointers()
display_proxy.set_camera(dp.f3(0, 0.06, 0.4), dp.f3(0, 0.06, 0))
colormap_tex = display_proxy.create_colormap_viridis()

# display_proxy.add_map_graphical_pointers(dp)
# display_proxy.add_step(solver, 10)
# display_proxy.add_normalize(solver, solver.particle_v,
#                             particle_normalized_attr, 0, 2)
# display_proxy.add_unmap_graphical_pointers(dp)
# display_proxy.add_particle_shading_program(solver.particle_x,
#                                            particle_normalized_attr,
#                                            colormap_tex,
#                                            solver.particle_radius, solver)
# display_proxy.add_pile_shading_program(pile)
#
# display_proxy.run()

display_proxy.add_particle_shading_program(solver.particle_x,
                                           particle_normalized_attr,
                                           colormap_tex,
                                           solver.particle_radius, solver)
display_proxy.add_pile_shading_program(pile)

with open('switch', 'w') as f:
    f.write('1')
while True:
    dp.map_graphical_pointers()
    with open('switch', 'r') as f:
        if f.read(1) == '0':
            solver.particle_x.write_file('rest-block-x.alu',
                                         solver.num_particles)
            solver.particle_v.write_file('rest-block-v.alu',
                                         solver.num_particles)
            pile.write_file('rest-block.pile')
            break
    for frame_interstep in range(100):
        # buoyant_force = runner.sum(solver.particle_force,
        #                            solver.particle_force.get_shape()[1],
        #                            solver.particle_force.get_shape()[1]).y
        # print(buoyant_force)
        # pile.x[1] = dp.f3(np.sin(solver.t *  2 * np.pi) * 0.03, container_width/2, 0)
        solver.step()

    solver.normalize(solver.particle_v, particle_normalized_attr, 0, 2)
    dp.unmap_graphical_pointers()
    display_proxy.draw()
