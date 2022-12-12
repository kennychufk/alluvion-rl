import alluvion as al
import numpy as np
import math
import argparse

from util import Unit, FluidSamplePellets, BuoySpec, read_pile

parser = argparse.ArgumentParser(description='Silhouette cutter')
parser.add_argument('--truth-dir', type=str, required=True)
parser.add_argument('--shape-dir', type=str, required=True)
parser.add_argument('--num-frames', type=int, default=1000)
args = parser.parse_args()

dp = al.Depot(np.float32)
cn = dp.cn
cni = dp.cni
runner = dp.Runner()

particle_radius = 0.25
kernel_radius = 1.0
density0 = 1.0
cubical_particle_volume = 8 * particle_radius * particle_radius * particle_radius
volume_relative_to_cube = 0.8
particle_mass = cubical_particle_volume * volume_relative_to_cube * density0
gravity = dp.f3(0, -1, 0)

density0_real = np.load(f'{args.truth_dir}/density0_real.npy')

unit = Unit(real_kernel_radius=2**-8,
            real_density0=density0_real,
            real_gravity=-9.80665)  # use real-world units

cn.set_kernel_radius(kernel_radius)
cn.set_particle_attr(particle_radius, particle_mass, density0)
cn.boundary_epsilon = 1e-9
cn.gravity = gravity

cni.max_num_particles_per_cell = 64
cni.max_num_neighbors_per_particle = 64

agitator_option = np.load(f'{args.truth_dir}/agitator_option.npy').item()
agitator_model_dir = f'{args.shape_dir}/{agitator_option}/models'
agitator_mesh_filename = f'{agitator_model_dir}/manifold2-decimate-2to-8.obj'
agitator_pellet_filename = f'{agitator_model_dir}/manifold2-decimate-2to-8.alu'

agitator_num_pellets = dp.get_alu_info(agitator_pellet_filename)[0][0]
num_pellets = agitator_num_pellets
agitator_exclusion_dist = unit.from_real_length(0.06)  # in real units

max_num_contacts = 2
pile = dp.Pile(dp, runner, max_num_contacts, al.VolumeMethod.pellets,
               num_pellets)

container_width = unit.from_real_length(0.24)
container_dim = dp.f3(container_width, container_width, container_width)
container_distance = dp.BoxDistance.create(container_dim, outset=0)
container_extent = container_distance.aabb_max - container_distance.aabb_min

agitator_mesh = al.Mesh()
agitator_mesh.set_obj(agitator_mesh_filename)
agitator_triangle_mesh = dp.TriangleMesh()
agitator_mesh.copy_to(agitator_triangle_mesh)
agitator_distance = dp.MeshDistance.create(agitator_triangle_mesh,
                                           offset=agitator_exclusion_dist)
agitator_id = pile.add_pellets(agitator_distance,
                               margin=0,
                               cell_width=kernel_radius)
print('agitator_extent', pile.domain_max_list[0] - pile.domain_min_list[0])
print('agitator_res', pile.resolution_list[0])
pile.reallocate_kinematics_on_device()
pile.set_gravity(gravity)

fluid_mass = np.load(f'{args.truth_dir}/fluid_mass.npy')
num_particles = int(fluid_mass / unit.to_real_mass(particle_mass))
print('num_particles', num_particles)

sampling = FluidSamplePellets(dp, f'{args.truth_dir}/sample-x.alu', cni)
sampling.sample_x.scale(unit.from_real_length(1))
container_aabb_range_per_h = container_extent / kernel_radius
cni.grid_res = al.uint3(int(math.ceil(container_aabb_range_per_h.x)),
                        int(math.ceil(container_aabb_range_per_h.y)),
                        int(math.ceil(container_aabb_range_per_h.z))) + 4
cni.grid_offset = al.int3(
    int(container_distance.aabb_min.x) - 2,
    int(container_distance.aabb_min.y) - 2,
    int(container_distance.aabb_min.z) - 2)

density_weight = dp.create_coated((sampling.num_samples), 1)

for frame_id in range(args.num_frames):
    xs, vs, qs, omegas = read_pile(f'{args.truth_dir}/{frame_id}.pile')
    pile.x[0] = unit.from_real_length(dp.f3(xs[-1]))
    pile.q[0] = dp.f4(qs[-1])
    pile.copy_kinematics_to_device()
    density_weight.fill(1)
    pile.compute_mask(0, 0, sampling.sample_x, density_weight,
                      sampling.num_samples)
    density_weight.write_file(
        f'{args.truth_dir}/density-weight-{frame_id}.alu',
        sampling.num_samples)
