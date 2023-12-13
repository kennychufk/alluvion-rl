import alluvion as al
import numpy as np
import argparse

from util import Unit, FluidSample, BuoySpec

parser = argparse.ArgumentParser(description='Silhouette cutter')
parser.add_argument('--truth-dir', type=str, required=True)
parser.add_argument('--shape-dir', type=str, required=True)
parser.add_argument('--num-frames', type=int, default=1000)
args = parser.parse_args()

unit = Unit(real_kernel_radius=1, real_density0=1,
            real_gravity=-1)  # use real-world units
agitator_exclusion_dist = 0.06  # in real units
buoy_exclusion_dist = 0.01
original_kernel_radius = 2**-8  # NOTE: for scaling agitator

dp = al.Depot(np.float32)
runner = dp.Runner()
max_num_contacts = 2
pile = dp.Pile(dp, runner, max_num_contacts)
num_buoys = dp.Pile.get_size_from_file(f'{args.truth_dir}/0.pile') - 2

pile.add(dp.BoxDistance.create(dp.f3(1, 1, 1)),
         al.uint3(1, 1, 1),
         sign=-1,
         mass=0)  # a placeholder for the container

buoy = BuoySpec(dp, unit)

for i in range(num_buoys):
    buoy_id = pile.add(
        buoy.create_distance(
            inset=-buoy_exclusion_dist),  # NOTE: remove distance offset
        al.uint3(24, 153, 24),
        sign=1,
        collision_mesh=buoy.mesh,
        mass=buoy.mass,
        inertia_tensor=buoy.inertia)

agitator_mesh = al.Mesh()
agitator_option = np.load(f'{args.truth_dir}/agitator_option.npy').item()
agitator_model_dir = f'{args.shape_dir}/{agitator_option}/models'
agitator_mesh_filename = f'{agitator_model_dir}/manifold2-decimate-2to-8.obj'
agitator_mesh.set_obj(agitator_mesh_filename)
agitator_mesh.scale(
    original_kernel_radius)  # NOTE: the loaded scale is not in real units
agitator_triangle_mesh = dp.TriangleMesh()
agitator_mesh.copy_to(agitator_triangle_mesh)
agitator_distance = dp.MeshDistance.create(
    agitator_triangle_mesh,
    offset=agitator_exclusion_dist)  # NOTE: remove distance offset
agitator_extent = agitator_distance.aabb_max - agitator_distance.aabb_min
agitator_res_float = agitator_extent / 0.0048  # NOTE: does not need to be the same resolution with the original
print('agitator_extent', agitator_extent)
agitator_res = al.uint3(int(agitator_res_float.x), int(agitator_res_float.y),
                        int(agitator_res_float.z))
print('agitator_res', agitator_res)
agitator_id = pile.add(agitator_distance,
                       agitator_res,
                       sign=1,
                       collision_mesh=agitator_mesh,
                       mass=1,
                       restitution=0.8,
                       friction=0.3,
                       display_mesh=agitator_mesh)

sampling = FluidSample(dp, f'{args.truth_dir}/sample-x.alu')
mask = dp.create_coated((sampling.num_samples), 1)
zero3 = dp.create_coated_like(sampling.sample_data3)
zero3.set_zero()

sum_v_norm_sqr = 0
max_v_norm_sqr = 0
sum_v_norm = 0
max_v_norm = 0
for frame_id in range(args.num_frames):
    pile.read_file(f'{args.truth_dir}/{frame_id}.pile')
    sampling.sample_data1.read_file(f'{args.truth_dir}/density-{frame_id}.alu')
    sampling.sample_data3.read_file(f'{args.truth_dir}/v-{frame_id}.alu')
    mask.fill(1)
    pile.compute_mask(num_buoys + 1, 0, sampling.sample_x, mask,
                      sampling.num_samples)

    mask.write_file(f'{args.truth_dir}/mask-{frame_id}.ver1.alu',
                    sampling.num_samples)
