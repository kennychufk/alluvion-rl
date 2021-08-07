import alluvion as al
import numpy as np
import argparse

from pathlib import Path


class FluidField:
    def __init__(self, dp, field_box_min, field_box_max):
        self.dp = dp
        self.num_samples_per_dim = 24
        self.num_samples = self.num_samples_per_dim * self.num_samples_per_dim * self.num_samples_per_dim
        self.sample_x = dp.create_coated((self.num_samples), 3)
        self.sample_data3 = dp.create_coated((self.num_samples), 3)
        self.sample_neighbors = dp.create_coated(
            (self.num_samples, dp.cni.max_num_neighbors_per_particle), 4)
        self.sample_num_neighbors = dp.create_coated((self.num_samples), 1,
                                                     np.uint32)

        self.sample_x_host = np.zeros((self.num_samples, 3), dp.default_dtype)

        field_box_size = dp.f3(field_box_max.x - field_box_min.x,
                               field_box_max.y - field_box_min.y,
                               field_box_max.z - field_box_min.z)
        for i in range(self.num_samples):
            z_id = i % self.num_samples_per_dim
            y_id = i % (self.num_samples_per_dim *
                        self.num_samples_per_dim) // self.num_samples_per_dim
            x_id = i // (self.num_samples_per_dim * self.num_samples_per_dim)
            self.sample_x_host[i] = np.array([
                field_box_min.x + field_box_size.x /
                (self.num_samples_per_dim - 1) * x_id, field_box_min.y +
                field_box_size.y / (self.num_samples_per_dim - 1) * y_id,
                field_box_min.z + field_box_size.z /
                (self.num_samples_per_dim - 1) * z_id
            ])
        self.sample_x.set(self.sample_x_host)


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

# rigids
max_num_contacts = 512
pile = dp.Pile(dp, max_num_contacts)

container_width = 0.24
container_dim = dp.f3(container_width, container_width, container_width)

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
solver.max_dt = particle_radius * 0.2
solver.min_dt = 0.0
solver.cfl = 0.15
solver.particle_radius = particle_radius
solver.num_particles = max_num_particles

dp.copy_cn()

display_proxy.set_camera(al.float3(0, 0.06, 0.4), al.float3(0, 0.06, 0))
colormap_tex = display_proxy.create_colormap_viridis()

display_proxy.add_particle_shading_program(solver.particle_x,
                                           particle_normalized_attr,
                                           colormap_tex,
                                           solver.particle_radius, solver)

field_box_min = dp.f3(container_width * -0.5, 0, container_width * -0.5)
field_box_max = dp.f3(container_width * 0.5, container_width,
                      container_width * 0.5)
field = FluidField(dp, field_box_min, field_box_max)

# frame_id = 0
# while True:
for frame_id in range(1000):
    dp.map_graphical_pointers()
    solver.num_particles = solver.particle_x.read_file(
        f'{args.input}/x-{frame_id}.alu')
    solver.particle_v.read_file(f'{args.input}/v-{frame_id}.alu')
    solver.normalize(solver.particle_v, particle_normalized_attr, 0, 2)

    solver.update_particle_neighbors()
    runner.launch_make_neighbor_list(field.sample_x, solver.pid,
                                     solver.pid_length, field.sample_neighbors,
                                     field.sample_num_neighbors,
                                     field.num_samples)
    runner.launch_sample_fluid(field.sample_x, solver.particle_x,
                               solver.particle_density, solver.particle_v,
                               field.sample_neighbors,
                               field.sample_num_neighbors, field.sample_data3,
                               field.num_samples)
    field.sample_data3.write_file(f"{args.input}/vfield-{frame_id}.alu",
                                  field.num_samples)

    dp.unmap_graphical_pointers()
    display_proxy.draw()
    print(frame_id)
    # frame_id += 1
