import math
import time

import alluvion as al
import numpy as np
from scipy.spatial.transform import Rotation as R

from util import Unit, BuoySpec

dp = al.Depot(np.float32)
cn = dp.cn
cni = dp.cni
dp.create_display(800, 600, "", False)
display_proxy = dp.get_display_proxy()
runner = dp.Runner()
particle_radius = 0.25
kernel_radius = 1.0
density0 = 1.0
cubical_particle_volume = 8 * particle_radius * particle_radius * particle_radius
volume_relative_to_cube = 0.8
particle_mass = cubical_particle_volume * volume_relative_to_cube * density0

gravity = dp.f3(0, -1, 0)

# real_kernel_radius = 0.0025
unit = Unit(real_kernel_radius=0.005,
            real_density0=1000,
            real_gravity=-9.80665)

cn.set_kernel_radius(kernel_radius)
cn.set_particle_attr(particle_radius, particle_mass, density0)
cn.gravity = gravity
cn.viscosity, cn.boundary_viscosity = unit.from_real_kinematic_viscosity(
    np.array([2.049e-6, 6.532e-6]))

cn.inertia_inverse = 0.5
cn.vorticity_coeff = 0.01
cn.viscosity_omega = 0.1

# rigids
max_num_contacts = 512
pile = dp.Pile(dp, runner, max_num_contacts)

target_container_volume = unit.from_real_volume(0.01)
container_mesh = al.Mesh()
# container_filename = '/media/kennychufk/Data/KennyCHU/ShapeNetCore.v2/02880940/b5d81a5bbbb8efe7c785f06f424b9d06/models/inv-meshfix.obj' # OK
# container_filename = '/media/kennychufk/Data/KennyCHU/ShapeNetCore.v2/02880940/a593e8863200fdb0664b3b9b23ddfcbc/models/inv-meshfix.obj' #OK
# container_filename = '/media/kennychufk/Data/KennyCHU/ShapeNetCore.v2/03991062/3fd59dd13de9ccfd703ecb6aac9c5d3c/models/inv-meshfix.obj' #OK
# container_filename = '/media/kennychufk/Data/KennyCHU/ShapeNetCore.v2/03991062/be0b5e9deced304a2c59a4d90e63212/models/inv-meshfix.obj' #OK
# container_filename = '/media/kennychufk/Data/KennyCHU/ShapeNetCore.v2/03991062/3152bd6fb7bf09d625ebd1cd0b422e32/models/inv-meshfix.obj' # OK except one particle
# container_filename = '/media/kennychufk/Data/KennyCHU/ShapeNetCore.v2/02808440/e78f7e89878402513af30a3946d92feb/models/inv-meshfix.obj' #OK
# container_filename = '/media/kennychufk/Data/KennyCHU/ShapeNetCore.v2/02808440/22eb6e2af43e6e78ba47eca8aa80094/models/inv-meshfix.obj' # OK
container_filename = '/media/kennychufk/Data/KennyCHU/ShapeNetCore.v2/02808440/549663876b12914accd0340812259a39/models/inv-meshfix.obj'  # OK
# container_filename = '/media/kennychufk/Data/KennyCHU/ShapeNetCore.v2/02808440/c4c3e4b9223ac6b71c17aef130ed6213/models/inv-meshfix.obj' # OK
# container_filename = '/media/kennychufk/Data/KennyCHU/ShapeNetCore.v2/03593526/6c7bcc925e7c6fdfb33495457ee36a49/models/inv-meshfix.obj' #OK
# container_filename = '/media/kennychufk/Data/KennyCHU/ShapeNetCore.v2/02876657/35ad544a3a4b25b122de9a89c84a71b7/models/inv-meshfix.obj' #OK
container_mesh.set_obj(container_filename)
original_volume, _, _, _ = container_mesh.calculate_mass_properties()
print(original_volume)
container_scale = np.cbrt(target_container_volume / original_volume)
container_mesh.scale(container_scale)
new_volume, _, _, _ = container_mesh.calculate_mass_properties()
print(new_volume)

container_triangle_mesh = dp.TriangleMesh()
container_mesh.copy_to(container_triangle_mesh)
container_distance = dp.MeshDistance.create(container_triangle_mesh)

print(container_distance.aabb_min)
print(container_distance.aabb_max)

container_extent = container_distance.aabb_max - container_distance.aabb_min
container_res_float = container_extent / 1
container_res = al.uint3(int(container_res_float.x),
                         int(container_res_float.y),
                         int(container_res_float.z))
print('container_res', container_res)
pile.add(container_distance,
         container_res,
         sign=-1,
         collision_mesh=container_mesh,
         mass=0,
         restitution=0.8,
         friction=0.3)

buoy = BuoySpec(dp, unit)
num_buoys = 8


def get_random_position(aabb_min, aabb_max):
    return dp.f3(np.random.uniform(aabb_min.x, aabb_max.x),
                 np.random.uniform(aabb_min.y, aabb_max.y),
                 np.random.uniform(aabb_min.z, aabb_max.z))


def get_random_quat():
    scipy_quat = R.random().as_quat()
    return dp.f4(scipy_quat[0], scipy_quat[1], scipy_quat[2], scipy_quat[3])


def has_collision(pile, i):
    num_contacts = pile.find_contacts()
    if (num_contacts > 0):
        print(i, 'contains collision', num_contacts)
        print('pile 0 x', pile.x[0])
    return num_contacts > 0
    # for j in range(pile.get_size()):
    #     if j != i:
    #         num_contacts1 = pile.find_contacts(i, j)
    #         num_contacts1a = dp.coat(pile.num_contacts).get()[0]
    #         num_contacts2 = pile.find_contacts(j, i)
    #         num_contacts2a = dp.coat(pile.num_contacts).get()[0]
    #         if num_contacts1>0 or num_contacts2>0:
    #             print("contains collision", i, j, num_contacts1, num_contacts2, num_contacts1a, num_contacts2a)
    #             # print("contains collision", i, j, pile.find_contacts(i, j), num_contacts2)
    #             return True
    # return False


def has_collision2(pile, i):
    for j in range(pile.get_size()):
        if j != i:
            num_contacts2 = pile.find_contacts(j, i)
            if num_contacts2 > 0:
                print("contains collision", i, j, num_contacts2)
                return True
    return False


for i in range(num_buoys):
    buoy_id = pile.add(buoy.create_distance(),
                       buoy.map_dim,
                       sign=1,
                       collision_mesh=buoy.mesh,
                       mass=buoy.mass,
                       restitution=0.3,
                       friction=0.4,
                       inertia_tensor=buoy.inertia,
                       x=get_random_position(container_distance.aabb_min,
                                             container_distance.aabb_max),
                       q=dp.f4(0, 0, 0, 1),
                       display_mesh=buoy.mesh)
    while (has_collision(pile, buoy_id)):
        pile.x[buoy_id] = get_random_position(container_distance.aabb_min,
                                              container_distance.aabb_max)

agitator_mesh = al.Mesh()
# agitator_filename = '3dmodels/bunny-pa.obj'
# agitator_filename = '/media/kennychufk/Data/KennyCHU/ShapeNetCore.v2/03797390/ec846432f3ebedf0a6f32a8797e3b9e9/models/manifold2-decimate.obj'
# agitator_filename = '/media/kennychufk/Data/KennyCHU/ShapeNetCore.v2/03046257/757fd88d3ddca2403406473757712946/models/manifold2-decimate.obj'
# agitator_filename = '/media/kennychufk/Data/KennyCHU/ShapeNetCore.v2/02942699/9db4b2c19c858a36eb34db531a289b8e/models/manifold2-decimate.obj'
# agitator_filename = '/media/kennychufk/Data/KennyCHU/ShapeNetCore.v2/03261776/1d4f9c324d6388a9b904f4192b538029/models/manifold2-decimate.obj'
# agitator_filename = '/media/kennychufk/Data/KennyCHU/ShapeNetCore.v2/03325088/daa26435e1603392b7a867e9b35a1295/models/manifold2-decimate.obj'
# agitator_filename = '/media/kennychufk/Data/KennyCHU/ShapeNetCore.v2/03759954/35f36e337df50fcb92b3c4741299c1af/models/manifold2-decimate.obj'
# agitator_filename = '/media/kennychufk/Data/KennyCHU/ShapeNetCore.v2/04401088/e5bb37415abcf3b7e1c1039f91f43fda/models/manifold2-decimate.obj'
agitator_filename = '/media/kennychufk/Data/KennyCHU/ShapeNetCore.v2/04530566/66a90b7b92ff2549f2635cfccf45023/models/manifold2-decimate.obj'
# agitator_filename = '/media/kennychufk/Data/KennyCHU/ShapeNetCore.v2/03513137/91c0193d38f0c5338c9affdacaf55648/models/manifold2-decimate.obj'
# agitator_filename = '/media/kennychufk/Data/KennyCHU/ShapeNetCore.v2/03636649/83353863ea1349682ebeb1e6a8111f53/models/manifold2-decimate.obj'
agitator_mesh.set_obj(agitator_filename)
agitator_original_vol, _, _, _ = agitator_mesh.calculate_mass_properties(1)
agitator_mesh.scale(np.cbrt(193 / agitator_original_vol))
agitator_density = unit.from_real_density(800)
agitator_mass, agitator_com, agitator_inertia, agitator_inertia_off_diag = agitator_mesh.calculate_mass_properties(
    agitator_density)
new_vol, _, _, _ = agitator_mesh.calculate_mass_properties(1)
print('new_vol', new_vol)
agitator_triangle_mesh = dp.TriangleMesh()
agitator_mesh.copy_to(agitator_triangle_mesh)
agitator_distance = dp.MeshDistance.create(agitator_triangle_mesh)
agitator_extent = agitator_distance.aabb_max - agitator_distance.aabb_min
print('agitator_extent', agitator_extent)
agitator_res_float = agitator_extent / particle_radius
agitator_res = al.uint3(int(agitator_res_float.x), int(agitator_res_float.y),
                        int(agitator_res_float.z))
num_trials = 0
agitator_id = pile.add(agitator_distance,
                       agitator_res,
                       sign=1,
                       collision_mesh=agitator_mesh,
                       mass=agitator_mass,
                       restitution=0.8,
                       friction=0.3,
                       inertia_tensor=agitator_inertia,
                       x=get_random_position(container_distance.aabb_min,
                                             container_distance.aabb_max),
                       q=get_random_quat(),
                       display_mesh=agitator_mesh)
while has_collision(pile, agitator_id):
    # pile.replace(agitator_id,
    #                 dp.MeshDistance.create(agitator_triangle_mesh),
    #                 agitator_res,
    #                 sign=1,
    #                 collision_mesh=agitator_mesh,
    #                 mass=agitator_mass,
    #                 restitution=0.8,
    #                 friction=0.3,
    #                 inertia_tensor=agitator_inertia,
    #                 x=get_random_position(container_distance.aabb_min, container_distance.aabb_max),
    #                 q=get_random_quat(),
    #                 display_mesh=agitator_mesh)
    pile.x[agitator_id] = get_random_position(container_distance.aabb_min,
                                              container_distance.aabb_max)
    print(pile.x[agitator_id])
    pile.q[agitator_id] = get_random_quat()
    print(pile.q[agitator_id])
    num_trials += 1
    if (num_trials > 100):
        import sys
        sys.exit(0)

pile.reallocate_kinematics_on_device()
pile.set_gravity(gravity)
cn.contact_tolerance = particle_radius

fluid_block_mode = 0
num_positions = dp.Runner.get_fluid_block_num_particles(
    mode=fluid_block_mode,
    box_min=container_distance.aabb_min,
    box_max=container_distance.aabb_max,
    particle_radius=particle_radius)
internal_encoded = dp.create_coated((num_positions), 1, np.uint32)
max_fill_num_particles = pile.compute_sort_fluid_block_internal_all(
    internal_encoded,
    box_min=container_distance.aabb_min,
    box_max=container_distance.aabb_max,
    particle_radius=particle_radius,
    mode=fluid_block_mode)
num_particles = 320000
# num_particles = int(max_fill_num_particles*0.6)
# num_particles = max_fill_num_particles
print('num_positions', num_positions)
print('num_particles', num_particles)

container_aabb_range_per_h = container_extent / kernel_radius
grid_res = al.uint3(int(math.ceil(container_aabb_range_per_h.x)),
                    int(math.ceil(container_aabb_range_per_h.y)),
                    int(math.ceil(container_aabb_range_per_h.z))) + 4
grid_offset = al.int3(
    int(container_distance.aabb_min.x) - 2,
    int(container_distance.aabb_min.y) - 2,
    int(container_distance.aabb_min.z) - 2)
print('grid_res', grid_res)
print('grid_offset', grid_offset)

cni.grid_res = grid_res
cni.grid_offset = grid_offset
cni.max_num_particles_per_cell = 64
cni.max_num_neighbors_per_particle = 64

solver = dp.SolverI(runner,
                    pile,
                    dp,
                    num_particles,
                    num_ushers=0,
                    enable_surface_tension=False,
                    enable_vorticity=False,
                    graphical=True)
particle_normalized_attr = dp.create_graphical_like(solver.particle_density)
solver.num_particles = num_particles
solver.max_dt = unit.from_real_time(0.1 * unit.rl)
solver.initial_dt = solver.max_dt
solver.min_dt = 0
solver.cfl = 0.4
# solver.density_error_tolerance = 1e-4

dp.map_graphical_pointers()
runner.launch_create_fluid_block_internal(solver.particle_x,
                                          internal_encoded,
                                          num_particles,
                                          offset=0,
                                          particle_radius=particle_radius,
                                          mode=fluid_block_mode,
                                          box_min=container_distance.aabb_min,
                                          box_max=container_distance.aabb_max)
dp.unmap_graphical_pointers()

display_proxy.set_camera(unit.from_real_length(al.float3(0, 0.6, 0.6)),
                         unit.from_real_length(al.float3(0, -0.02, 0)))
display_proxy.set_clip_planes(unit.to_real_length(1),
                              container_distance.aabb_max.z * 20)
colormap_tex = display_proxy.create_colormap_viridis()

display_proxy.add_particle_shading_program(solver.particle_x,
                                           particle_normalized_attr,
                                           colormap_tex,
                                           solver.particle_radius, solver)
display_proxy.add_pile_shading_program(pile)

#display_proxy.run()
for frame_id in range(2000000):
    display_proxy.draw()
    dp.map_graphical_pointers()
    for substep_id in range(20):
        solver.step()
    solver.normalize(solver.particle_v, particle_normalized_attr, 0,
                     unit.from_real_velocity(0.01))
    dp.unmap_graphical_pointers()

dp.remove(internal_encoded)
dp.remove(particle_normalized_attr)
