import alluvion as al
import sys
import numpy as np
from numpy import linalg as LA
import time

from util import BuoySpec, Unit

dp = al.Depot(np.float32)
cn = dp.cn
cni = dp.cni
dp.create_display(800, 600, "", False)
display_proxy = dp.get_display_proxy()
runner = dp.Runner()

density0 = 1
unit = Unit(real_kernel_radius=2**-8,
            real_density0=density0,
            real_gravity=-9.80665)

scale_factor = 1
kernel_radius = scale_factor
particle_radius = kernel_radius * 0.25
volume_relative_to_cube = 0.8
particle_mass = volume_relative_to_cube * kernel_radius * kernel_radius * kernel_radius * 0.125 * density0

cn.set_kernel_radius(kernel_radius)
cn.set_particle_attr(particle_radius, particle_mass, density0)
cn.contact_tolerance = 0

buoy = BuoySpec(dp, unit)

print("buoy radius", buoy.radius)
print("buoy map_dim", buoy.map_dim)

pile = dp.Pile(dp,
               runner,
               max_num_contacts=0,
               volume_method=al.VolumeMethod.pellets,
               max_num_pellets=0)
inset = unit.from_real_length(10e-5)
pile.add(buoy.create_distance(inset), buoy.map_dim, sign=-1)
pile.reallocate_kinematics_on_device()

# ========= sampling
domain_min = pile.domain_min_list[0]
domain_max = pile.domain_max_list[0]
grid_margin = 1
domain_min_multiples = domain_min
domain_max_multiples = domain_max
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

fill_particle_radius = particle_radius * 0.67
fill_margin = 1
fill_domain_min = domain_min - kernel_radius * fill_margin
fill_domain_max = domain_max + kernel_radius * fill_margin
print('fill_domain', fill_domain_min, fill_domain_max)
print('fill_domain extent', domain_max - domain_min)
print('fill_particle_radius', fill_particle_radius)
cn.set_kernel_radius(0)
dp.copy_cn()
num_sample_positions = dp.Runner.get_fluid_cylinder_num_particles(
    buoy.radius, fill_domain_min.y, fill_domain_max.y, fill_particle_radius)
print('num_sample_positions', num_sample_positions)
internal_encoded = dp.create((num_sample_positions), 1, np.uint32)
max_num_pellets = pile.compute_sort_fluid_cylinder_internal_all(
    internal_encoded, buoy.radius, fill_particle_radius, fill_domain_min.y,
    fill_domain_max.y)
max_num_pellets -= 40
print("num_pellets for filling", max_num_pellets)
x = dp.create_coated((max_num_pellets), 3)
runner.launch_create_fluid_cylinder_internal(x, internal_encoded,
                                             max_num_pellets, 0, buoy.radius,
                                             fill_particle_radius,
                                             fill_domain_min.y,
                                             fill_domain_max.y)
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
for fill_step_id in range(2000):
    display_proxy.draw()
    dp.map_graphical_pointers()
    for substep_id in range(10):
        solver.step()
    time.sleep(0.01)
    solver.normalize(solver.particle_v, pellet_normalized_attr, 0,
                     kernel_radius * 0.2)
    pellet_density = dp.coat(solver.particle_density).get(solver.num_particles)
    max_pellet_density = np.max(pellet_density)
    min_pellet_density = np.min(pellet_density)
    mean_pellet_density = np.mean(pellet_density)
    if mean_pellet_density > 1.3:
        solver.num_particles -= 1
    print('pellet_density', min_pellet_density, mean_pellet_density,
          max_pellet_density)
    dp.unmap_graphical_pointers()

dp.map_graphical_pointers()
solver.particle_x.write_file("buoy-2to-8.alu", solver.num_particles)
dp.unmap_graphical_pointers()
print('final num pellets', solver.num_particles)
