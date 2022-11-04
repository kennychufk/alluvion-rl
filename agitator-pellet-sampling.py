import alluvion as al
import sys
import numpy as np
from numpy import linalg as LA
import time

dp = al.Depot(np.float32)
cn = dp.cn
cni = dp.cni
dp.create_display(800, 600, "", False)
display_proxy = dp.get_display_proxy()
runner = dp.Runner()

scale_factor = 1
kernel_radius = scale_factor
particle_radius = kernel_radius * 0.25
density0 = 1
volume_relative_to_cube = 0.8
particle_mass = volume_relative_to_cube * kernel_radius * kernel_radius * kernel_radius * 0.125 * density0

cn.set_kernel_radius(kernel_radius)
cn.set_particle_attr(particle_radius, particle_mass, density0)
cn.contact_tolerance = 0

agitator_options = [
    'bunny/bunny', '03797390/ec846432f3ebedf0a6f32a8797e3b9e9',
    '03046257/757fd88d3ddca2403406473757712946',
    '02942699/9db4b2c19c858a36eb34db531a289b8e',
    '03261776/1d4f9c324d6388a9b904f4192b538029',
    '03325088/daa26435e1603392b7a867e9b35a1295',
    '03759954/35f36e337df50fcb92b3c4741299c1af',
    '04401088/e5bb37415abcf3b7e1c1039f91f43fda',
    '04530566/66a90b7b92ff2549f2635cfccf45023',
    '03513137/91c0193d38f0c5338c9affdacaf55648',
    '03636649/83353863ea1349682ebeb1e6a8111f53'
]
agitator_option = agitator_options[10]
# 6, 7, 8, 10 problematic
shape_dir = '/home/kennychufk/workspace/pythonWs/shape-al'
agitator_model_dir = f'{shape_dir}/{agitator_option}/models'
agitator_mesh_filename = f'{agitator_model_dir}/manifold2-decimate-2to-8.obj'

pile = dp.Pile(dp,
               runner,
               max_num_contacts=0,
               volume_method=al.VolumeMethod.pellets,
               max_num_pellets=0)
obj_mesh = al.Mesh()
obj_mesh.set_obj(agitator_mesh_filename)
obj_triangle_mesh = dp.TriangleMesh()
obj_mesh.copy_to(obj_triangle_mesh)
obj_mesh_distance = dp.MeshDistance.create(obj_triangle_mesh, offset=0)

# ========= sampling
domain_min = obj_mesh_distance.aabb_min
domain_max = obj_mesh_distance.aabb_max
grid_margin = 0
domain_min_multiples = domain_min / kernel_radius
domain_max_multiples = domain_max / kernel_radius
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
print('cni.grid_res', cni.grid_res)

pile.add(obj_mesh_distance, cni.grid_res * 4, sign=-1)
pile.reallocate_kinematics_on_device()

fill_mode = 2
fill_particle_radius = particle_radius * 0.98
fill_margin = 1
fill_domain_min = domain_min - kernel_radius * fill_margin
fill_domain_max = domain_max + kernel_radius * fill_margin
cn.set_kernel_radius(0)
dp.copy_cn()
num_sample_positions = dp.Runner.get_fluid_block_num_particles(
    fill_mode, fill_domain_min, fill_domain_max, fill_particle_radius)
print("num sample position", num_sample_positions)
internal_encoded = dp.create((num_sample_positions), 1, np.uint32)
max_num_pellets = pile.compute_sort_fluid_block_internal_all(
    internal_encoded, fill_domain_min, fill_domain_max, fill_particle_radius,
    fill_mode)
print("num_pellets for filling", max_num_pellets)
x = dp.create((max_num_pellets), 3)
runner.launch_create_fluid_block_internal(x, internal_encoded, max_num_pellets,
                                          0, fill_particle_radius, fill_mode,
                                          fill_domain_min, fill_domain_max)
cn.set_kernel_radius(kernel_radius)
dp.copy_cn()

pellet_normalized_attr = dp.create_graphical((max_num_pellets), 1)
solver = dp.SolverPellet(runner, pile, dp, max_num_pellets, graphical=True)
dp.map_graphical_pointers()
solver.set_pellets(x)
dp.unmap_graphical_pointers()
dp.remove(x)

domain_center = (domain_max + domain_min) * 0.5
domain_span = domain_max - domain_min
display_proxy.set_camera(domain_center + domain_max * 4, domain_center)
display_proxy.set_clip_planes(
    particle_radius * 0.1,
    np.max([domain_span.x, domain_span.y, domain_span.z]) * 10)
colormap_tex = display_proxy.create_colormap_viridis()
display_proxy.add_particle_shading_program(solver.particle_x,
                                           pellet_normalized_attr,
                                           colormap_tex,
                                           solver.particle_radius,
                                           solver,
                                           clear=True)

for fill_step_id in range(300):
    display_proxy.draw()
    dp.map_graphical_pointers()
    for substep_id in range(10):
        solver.step()
    time.sleep(0.01)
    solver.normalize(solver.particle_v, pellet_normalized_attr, 0,
                     kernel_radius * 0.2)
    pellet_density = dp.coat(
        solver.particle_density).get()[:solver.num_particles]
    max_pellet_density = np.max(pellet_density)
    min_pellet_density = np.min(pellet_density)
    mean_pellet_density = np.mean(pellet_density)
    print('pellet_density', min_pellet_density, mean_pellet_density,
          max_pellet_density)
    dp.unmap_graphical_pointers()

dp.map_graphical_pointers()
solver.particle_x.write_file(
    f'{agitator_model_dir}/manifold2-decimate-2to-8.alu', solver.num_particles)
with open(f'{agitator_model_dir}/manifold2-decimate-2to-8.fill-r.txt',
          'w') as f:
    f.write(f"{fill_particle_radius}")
dp.unmap_graphical_pointers()
