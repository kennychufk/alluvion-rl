import argparse
import math
from pathlib import Path
import alluvion as al
import numpy as np
from util import Unit, LeapInterpolator

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, required=True)
parser.add_argument('--shape-dir', type=str, required=True)

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

unit = Unit(real_kernel_radius=2**-8,
            real_density0=1000,
            real_gravity=-9.80665)

cn.set_kernel_radius(kernel_radius)
cn.set_particle_attr(particle_radius, particle_mass, density0)
cn.boundary_epsilon = 1e-9
cni.max_num_particles_per_cell = 64
cni.max_num_neighbors_per_particle = 64

agitator_options = [
    'bunny/bunny',
    '03797390/ec846432f3ebedf0a6f32a8797e3b9e9',
    '03046257/757fd88d3ddca2403406473757712946',
    '02942699/9db4b2c19c858a36eb34db531a289b8e',
    '03261776/1d4f9c324d6388a9b904f4192b538029',
    '03325088/daa26435e1603392b7a867e9b35a1295',
    '03513137/91c0193d38f0c5338c9affdacaf55648',
]

container_pellet_filename = f'{args.shape_dir}/cube24/cube24/models/cube24-2to-8.alu'
container_num_pellets = dp.get_alu_info(container_pellet_filename)[0][0]
num_pellets = container_num_pellets

for agitator_option in agitator_options:
    agitator_model_dir = f'{args.shape_dir}/{agitator_option}/models'
    agitator_pellet_filename = f'{agitator_model_dir}/manifold2-decimate-2to-8.alu'
    agitator_num_pellets = dp.get_alu_info(agitator_pellet_filename)[0][0]
    num_pellets += agitator_num_pellets

# rigids
max_num_contacts = 512
pile = dp.Pile(dp, runner, max_num_contacts, al.VolumeMethod.pellets,
               num_pellets)

## ================== using cube
container_width = unit.from_real_length(0.24)
container_dim = dp.f3(container_width, container_width, container_width)
container_distance = dp.BoxDistance.create(container_dim, outset=0)
container_extent = container_distance.aabb_max - container_distance.aabb_min
container_res_float = container_extent / particle_radius
container_res = al.uint3(int(container_res_float.x),
                         int(container_res_float.y),
                         int(container_res_float.z))
print('container_res', container_res)
container_pellet_x = dp.create((container_num_pellets), 3)
container_pellet_x.read_file(container_pellet_filename)
container_id = pile.add_pellets(container_distance,
                                container_res,
                                pellets=container_pellet_x,
                                sign=-1,
                                mass=0,
                                restitution=0.8,
                                friction=0.3)
dp.remove(container_pellet_x)
## ================== using cube


def has_collision(pile, i, j):
    return (pile.find_contacts(i, j) > 0 or pile.find_contacts(j, i) > 0)


interpolator = LeapInterpolator(dp, args.input)

agitator_ids = []
agitator_min_ys = [0.0]
agitator_max_ys = [0.0]
for agitator_option in agitator_options:
    agitator_mesh = al.Mesh()
    agitator_model_dir = f'{args.shape_dir}/{agitator_option}/models'
    agitator_mesh_filename = f'{agitator_model_dir}/manifold2-decimate-2to-8.obj'
    agitator_mesh.set_obj(agitator_mesh_filename)
    agitator_density_real = 1000
    agitator_density = unit.from_real_density(agitator_density_real)
    agitator_mass, agitator_com, agitator_inertia, agitator_inertia_off_diag = agitator_mesh.calculate_mass_properties(
        agitator_density)
    agitator_triangle_mesh = dp.TriangleMesh()
    agitator_mesh.copy_to(agitator_triangle_mesh)
    agitator_distance = dp.MeshDistance.create(agitator_triangle_mesh,
                                               +0.4 * kernel_radius)
    agitator_extent = agitator_distance.aabb_max - agitator_distance.aabb_min
    agitator_min_ys.append(agitator_distance.aabb_min.y)
    agitator_max_ys.append(agitator_distance.aabb_max.y)
    print('agitator_extent', agitator_extent)
    agitator_res_float = agitator_extent / particle_radius
    agitator_res = al.uint3(int(agitator_res_float.x),
                            int(agitator_res_float.y),
                            int(agitator_res_float.z))
    agitator_pellet_filename = f'{agitator_model_dir}/manifold2-decimate-2to-8.alu'
    agitator_num_pellets = dp.get_alu_info(agitator_pellet_filename)[0][0]
    agitator_pellet_x = dp.create((agitator_num_pellets), 3)
    agitator_pellet_x.read_file(agitator_pellet_filename)
    agitator_id = pile.add_pellets(agitator_distance,
                                   agitator_res,
                                   pellets=agitator_pellet_x,
                                   sign=1,
                                   mass=0,
                                   restitution=0.8,
                                   friction=0.3,
                                   inertia_tensor=agitator_inertia,
                                   display_mesh=agitator_mesh)
    agitator_ids.append(agitator_id)
    dp.remove(agitator_pellet_x)

pile.reallocate_kinematics_on_device()
pile.set_gravity(gravity)
cn.contact_tolerance = particle_radius * 2

container_aabb_range_per_h = container_extent / kernel_radius
cni.grid_res = al.uint3(int(math.ceil(container_aabb_range_per_h.x)),
                        int(math.ceil(container_aabb_range_per_h.y)),
                        int(math.ceil(container_aabb_range_per_h.z))) + 4
cni.grid_offset = al.int3(
    int(container_distance.aabb_min.x) - 2,
    int(container_distance.aabb_min.y) - 2,
    int(container_distance.aabb_min.z) - 2)
print('grid_res', cni.grid_res)
print('grid_offset', cni.grid_offset)

target_t = unit.from_real_time(10)
offset_y_interval = 0.001
tenth_second_frame_id = np.argmax(interpolator.t > 10)
for agitator_id in agitator_ids:
    offset_y_min = 0.24 * -0.5 - np.min(
        interpolator.x[:tenth_second_frame_id, 1]) - unit.to_real_length(
            agitator_min_ys[agitator_id])
    offset_y_max = 0.24 * 0.5 - np.max(
        interpolator.x[:tenth_second_frame_id, 1]) - unit.to_real_length(
            agitator_max_ys[agitator_id])
    print('offset_y_min', offset_y_min)
    print('offset_y_max', offset_y_max)
    for offset_y in np.arange(offset_y_min, offset_y_max, offset_y_interval):
        collided = False
        for t in np.arange(0.0, target_t, unit.from_real_time(10)):
            pile.x[agitator_id] = unit.from_real_length(
                interpolator.get_x(unit.to_real_time(t)) +
                dp.f3(0, offset_y, 0))
            pile.q[agitator_id] = dp.f4(
                interpolator.get_q(unit.to_real_time(t)))
            pile.copy_kinematics_to_device()
            if has_collision(pile, container_id, agitator_id):
                # print(f'Agitator {agitator_id} has collision', pile.x[agitator_id])
                collided = True
                break
        if not collided:
            input_path = Path(args.input)
            offset_dir = input_path.parent.joinpath("offsets")
            offset_dir.mkdir(exist_ok=True)
            agitator_postfix = agitator_options[agitator_id - 1].replace(
                '/', '-')
            offset_path = offset_dir.joinpath(
                f'{input_path.stem}.{agitator_postfix}.npy')
            print(offset_path)
            np.save(str(offset_path), np.array([0, offset_y, 0]))
            break
