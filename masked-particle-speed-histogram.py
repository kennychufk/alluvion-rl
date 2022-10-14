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

sampling = FluidSamplePellets(dp, f'{args.truth_dir}/x-0.alu', cni)
sampling.sample_x.scale(unit.from_real_length(1))
container_aabb_range_per_h = container_extent / kernel_radius
cni.grid_res = al.uint3(int(math.ceil(container_aabb_range_per_h.x)),
                        int(math.ceil(container_aabb_range_per_h.y)),
                        int(math.ceil(container_aabb_range_per_h.z))) + 4
cni.grid_offset = al.int3(
    int(container_distance.aabb_min.x) - 2,
    int(container_distance.aabb_min.y) - 2,
    int(container_distance.aabb_min.z) - 2)

solver = dp.SolverI(runner,
                    pile,
                    dp,
                    num_particles,
                    enable_surface_tension=False,
                    enable_vorticity=False)

solver.num_particles = num_particles
solver.max_dt = unit.from_real_time(0.00025)
solver.initial_dt = solver.max_dt
solver.min_dt = 0
solver.cfl = 0.2
solver.min_density_solve = 5

max_v_bin = 0.25

partial_histogram = dp.create_coated((al.kPartialHistogram256Size), 1,
                                     np.uint32)
histogram = dp.create_coated((al.kHistogram256BinCount), 1, np.uint32)
v = dp.create_coated((sampling.num_samples), 1)
mask = dp.create_coated((sampling.num_samples), 1)
quantized4s = dp.create_coated(((sampling.num_samples - 1) // 4 + 1), 1,
                               np.uint32)
histograms = np.zeros((args.num_frames, al.kHistogram256BinCount), np.uint32)

for frame_id in range(args.num_frames):
    xs, vs, qs, omegas = read_pile(f'{args.truth_dir}/{frame_id}.pile')
    pile.x[0] = unit.from_real_length(dp.f3(xs[-1]))
    pile.q[0] = dp.f4(qs[-1])
    pile.copy_kinematics_to_device()
    sampling.sample_x.read_file(f'{args.truth_dir}/x-{frame_id}.alu')
    sampling.sample_x.scale(unit.from_real_length(1))
    mask.fill(1)
    pile.compute_mask(0, 0, sampling.sample_x, mask, sampling.num_samples)
    v.read_file(f'{args.truth_dir}/v2-{frame_id}.alu')
    runner.sqrt_inplace(v, v.get_linear_shape())
    runner.launch_histogram256_with_mask(partial_histogram, histogram,
                                         quantized4s, mask, v, 0, max_v_bin,
                                         sampling.num_samples)
    histograms[frame_id] = histogram.get()
np.save(f'{args.truth_dir}/v-hist-masked.npy', histograms)
