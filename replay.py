import alluvion as al
import numpy as np
import argparse

from pathlib import Path
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

cn.set_kernel_radius(kernel_radius)
cn.set_particle_attr(particle_radius, particle_mass, density0)

# rigids
max_num_contacts = 512
pile = dp.Pile(dp, max_num_contacts)

max_num_particles = 5000000
grid_res = al.uint3(1, 1, 1)

solver = dp.Solver(runner,
                   pile,
                   dp,
                   max_num_particles,
                   enable_surface_tension=False,
                   enable_vorticity=False,
                   graphical=True)
particle_normalized_attr = dp.create_graphical((max_num_particles), 1)

display_proxy.set_camera(al.float3(0, 0.06, 0.4), al.float3(0, 0.06, 0))
colormap_tex = display_proxy.create_colormap_viridis()

display_proxy.add_particle_shading_program(solver.particle_x,
                                           particle_normalized_attr,
                                           colormap_tex,
                                           solver.particle_radius, solver)

# frame_id = 0
# while True:
for frame_id in range(1000):
    dp.map_graphical_pointers()
    solver.num_particles = solver.particle_x.read_file(
        f'{args.input}/x-{frame_id}.alu')
    solver.particle_v.read_file(f'{args.input}/v-{frame_id}.alu')
    solver.normalize(solver.particle_v, particle_normalized_attr, 0, 2)
    dp.unmap_graphical_pointers()
    display_proxy.draw()
    print(frame_id)
    # frame_id += 1

while True:
    display_proxy.draw()
