from pathlib import Path
import argparse
import math
import time
import numpy as np
import subprocess
import os
from scipy.spatial.transform import Rotation as R
from PIL import Image

import alluvion as al

from util import Unit, FluidSamplePellets, get_timestamp_and_hash, BuoySpec, parameterize_kinematic_viscosity

parser = argparse.ArgumentParser(description='RL ground truth generator')
parser.add_argument('--output-dir', type=str, default='.')
parser.add_argument('--write-visual', type=bool, default=False)
parser.add_argument('--render', type=bool, default=False)
parser.add_argument('--shape-dir', type=str, required=True)
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--num-buoys', type=int, required=True)
args = parser.parse_args()
np.random.seed(args.seed)
dp = al.Depot(np.float32)
cn = dp.cn
cni = dp.cni
dp.create_display(800, 600, "alluvion-fixed", False)
display_proxy = dp.get_display_proxy()
framebuffer = display_proxy.create_framebuffer()
runner = dp.Runner()

particle_radius = 0.25
kernel_radius = 1.0
density0 = 1.0
cubical_particle_volume = 8 * particle_radius * particle_radius * particle_radius
volume_relative_to_cube = 0.8
particle_mass = cubical_particle_volume * volume_relative_to_cube * density0

gravity = dp.f3(0, -1, 0)

density0_real = np.random.uniform(low=990, high=1200)
unit = Unit(real_kernel_radius=2**-8,
            real_density0=density0_real,
            real_gravity=-9.80665)

cn.set_kernel_radius(kernel_radius)
cn.set_particle_attr(particle_radius, particle_mass, density0)
cn.boundary_epsilon = 1e-9
cn.gravity = gravity
kinematic_viscosity_real = 1e-6  # TODO: discrete, either water or GWM
cn.viscosity, cn.boundary_viscosity = unit.from_real_kinematic_viscosity(
    parameterize_kinematic_viscosity(kinematic_viscosity_real))
print('parameterized nu',
      parameterize_kinematic_viscosity(kinematic_viscosity_real)[0])
cni.max_num_particles_per_cell = 64
cni.max_num_neighbors_per_particle = 64

agitator_options = [
    'bunny/bunny',
    '03797390/ec846432f3ebedf0a6f32a8797e3b9e9',
    '03046257/757fd88d3ddca2403406473757712946',
    '02942699/9db4b2c19c858a36eb34db531a289b8e',
    '03261776/1d4f9c324d6388a9b904f4192b538029',
    '03325088/daa26435e1603392b7a867e9b35a1295',
    '03759954/35f36e337df50fcb92b3c4741299c1af',
    # '04401088/e5bb37415abcf3b7e1c1039f91f43fda',
    # '04530566/66a90b7b92ff2549f2635cfccf45023',
    # '03513137/91c0193d38f0c5338c9affdacaf55648',
    # '03636649/83353863ea1349682ebeb1e6a8111f53'
]

agitator_selected_id = 0
agitator_option = agitator_options[agitator_selected_id]

agitator_model_dir = f'{args.shape_dir}/{agitator_option}/models'
agitator_mesh_filename = f'{agitator_model_dir}/manifold2-decimate-2to-8.obj'

container_pellet_filename = '/home/kennychufk/workspace/pythonWs/alluvion-optim/cube24-2to-8.alu'
buoy_pellet_filename = '/home/kennychufk/workspace/pythonWs/alluvion-optim/buoy-2to-8.alu'
agitator_pellet_filename = f'{agitator_model_dir}/manifold2-decimate-2to-8.alu'

num_buoys = args.num_buoys
container_num_pellets = dp.get_alu_info(container_pellet_filename)[0][0]
buoy_num_pellets = dp.get_alu_info(buoy_pellet_filename)[0][0]
agitator_num_pellets = dp.get_alu_info(agitator_pellet_filename)[0][0]
num_pellets = container_num_pellets + buoy_num_pellets * num_buoys + agitator_num_pellets

# rigids
max_num_contacts = 512
pile = dp.Pile(dp, runner, max_num_contacts, al.VolumeMethod.pellets,
               num_pellets)

target_container_volume = unit.from_real_volume(0.008)
container_mesh = al.Mesh()

## ================== using cube
container_width = unit.from_real_length(0.24)
container_dim = dp.f3(container_width, container_width, container_width)
container_mesh = al.Mesh()
container_mesh.set_box(container_dim, 8)
container_distance = dp.BoxDistance.create(container_dim, outset=0)
container_extent = container_distance.aabb_max - container_distance.aabb_min
container_res_float = container_extent / particle_radius
container_res = al.uint3(int(container_res_float.x),
                         int(container_res_float.y),
                         int(container_res_float.z))
print('container_res', container_res)
container_pellet_x = dp.create((container_num_pellets), 3)
container_pellet_x.read_file(container_pellet_filename)
pile.add_pellets(container_distance,
                 container_res,
                 pellets=container_pellet_x,
                 sign=-1,
                 mass=0,
                 restitution=0.8,
                 friction=0.3)
dp.remove(container_pellet_x)
## ================== using cube

buoy = BuoySpec(dp, unit)


def get_random_position(aabb_min, aabb_max):
    return dp.f3(np.random.uniform(aabb_min.x, aabb_max.x),
                 np.random.uniform(aabb_min.y, aabb_max.y),
                 np.random.uniform(aabb_min.z, aabb_max.z))


def get_random_quat():
    scipy_quat = R.random().as_quat()
    return dp.f4(scipy_quat[0], scipy_quat[1], scipy_quat[2], scipy_quat[3])


def has_collision(pile, i):
    for j in range(pile.get_size()):
        if j != i:
            if pile.find_contacts(i, j) > 0 or pile.find_contacts(j, i) > 0:
                print("contains collision", i, j)
                return True
    return False


buoy_pellet_x = dp.create((buoy_num_pellets), 3)
buoy_pellet_x.read_file(buoy_pellet_filename)
for i in range(num_buoys):
    buoy_id = pile.add_pellets(buoy.create_distance(0),
                               buoy.map_dim,
                               pellets=buoy_pellet_x,
                               sign=1,
                               mass=buoy.mass,
                               restitution=0.3,
                               friction=0.4,
                               inertia_tensor=buoy.inertia,
                               x=get_random_position(
                                   container_distance.aabb_min,
                                   container_distance.aabb_max),
                               q=dp.f4(0, 0, 0, 1),
                               display_mesh=buoy.mesh)
    while (has_collision(pile, buoy_id)):
        print('has collision', pile.x[buoy_id])
        pile.x[buoy_id] = get_random_position(container_distance.aabb_min,
                                              container_distance.aabb_max)
for i in range(num_buoys):
    print(f"{i+1}: {pile.x[i+1]}")
dp.remove(buoy_pellet_x)

agitator_mesh = al.Mesh()

print(agitator_mesh_filename)
print(Path(agitator_mesh_filename).is_file())
agitator_mesh.set_obj(agitator_mesh_filename)
agitator_density_real = 1000
agitator_density = unit.from_real_density(agitator_density_real)
agitator_mass, agitator_com, agitator_inertia, agitator_inertia_off_diag = agitator_mesh.calculate_mass_properties(
    agitator_density)
agitator_vol, _, _, _ = agitator_mesh.calculate_mass_properties(1)
print('agitator_vol real', unit.to_real_volume(agitator_vol))
agitator_triangle_mesh = dp.TriangleMesh()
agitator_mesh.copy_to(agitator_triangle_mesh)
agitator_distance = dp.MeshDistance.create(agitator_triangle_mesh,
                                           +0.4 * kernel_radius)
agitator_extent = agitator_distance.aabb_max - agitator_distance.aabb_min
print('agitator_extent', agitator_extent)
agitator_res_float = agitator_extent / particle_radius
agitator_res = al.uint3(int(agitator_res_float.x), int(agitator_res_float.y),
                        int(agitator_res_float.z))
agitator_pellet_x = dp.create((agitator_num_pellets), 3)
agitator_pellet_x.read_file(agitator_pellet_filename)
agitator_id = pile.add_pellets(agitator_distance,
                               agitator_res,
                               pellets=agitator_pellet_x,
                               sign=1,
                               mass=agitator_mass,
                               restitution=0.8,
                               friction=0.3,
                               inertia_tensor=agitator_inertia,
                               x=get_random_position(
                                   container_distance.aabb_min,
                                   container_distance.aabb_max),
                               q=get_random_quat(),
                               display_mesh=agitator_mesh)
dp.remove(agitator_pellet_x)
while has_collision(pile, agitator_id):
    pile.x[agitator_id] = get_random_position(container_distance.aabb_min,
                                              container_distance.aabb_max)
    pile.q[agitator_id] = get_random_quat()

pile.reallocate_kinematics_on_device()
pile.set_gravity(gravity)
cn.contact_tolerance = particle_radius * 2

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
# num_particles_options = [260000, 290000, 320000]
num_particles_options = [380000]
num_particles = num_particles_options[np.random.randint(
    low=0, high=len(num_particles_options))]
# num_particles = int(max_fill_num_particles*0.6)
# num_particles = max_fill_num_particles
print('num_positions', num_positions)
print('num_particles', num_particles)

## ======== internal sample points
sample_x_fill_mode = 0
sample_radius = kernel_radius
num_sample_positions = dp.Runner.get_fluid_block_num_particles(
    mode=sample_x_fill_mode,
    box_min=container_distance.aabb_min,
    box_max=container_distance.aabb_max,
    particle_radius=sample_radius)
sample_internal_encoded = dp.create_coated((num_sample_positions), 1,
                                           np.uint32)
num_samples = pile.compute_sort_fluid_block_internal_all(
    sample_internal_encoded,
    box_min=container_distance.aabb_min,
    box_max=container_distance.aabb_max,
    particle_radius=sample_radius,
    mode=sample_x_fill_mode)
print('num_samples', num_samples)
sampling = FluidSamplePellets(dp, np.zeros((num_samples, 3), dp.default_dtype))
runner.launch_create_fluid_block_internal(sampling.sample_x,
                                          sample_internal_encoded,
                                          num_samples,
                                          offset=0,
                                          particle_radius=sample_radius,
                                          mode=sample_x_fill_mode,
                                          box_min=container_distance.aabb_min,
                                          box_max=container_distance.aabb_max)
dp.remove(sample_internal_encoded)
## ======== end of internal sample points

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

solver = dp.SolverI(runner,
                    pile,
                    dp,
                    num_particles,
                    enable_surface_tension=False,
                    enable_vorticity=False,
                    graphical=True)
particle_normalized_attr = dp.create_graphical_like(solver.particle_density)

solver.num_particles = num_particles
solver.max_dt = unit.from_real_time(0.00025)
solver.initial_dt = solver.max_dt
solver.min_dt = 0
solver.cfl = 0.2
solver.min_density_solve = 5

dp.map_graphical_pointers()
runner.launch_create_fluid_block_internal(solver.particle_x,
                                          internal_encoded,
                                          num_particles,
                                          offset=0,
                                          particle_radius=particle_radius,
                                          mode=fluid_block_mode,
                                          box_min=container_distance.aabb_min,
                                          box_max=container_distance.aabb_max)
dp.remove(internal_encoded)
dp.unmap_graphical_pointers()

display_proxy.set_camera(unit.from_real_length(al.float3(0, 0.6, 0.6)),
                         unit.from_real_length(al.float3(0, -0.02, 0)))
display_proxy.set_clip_planes(unit.to_real_length(1),
                              container_distance.aabb_max.z * 20)
colormap_tex = display_proxy.create_colormap_viridis()

display_proxy.add_bind_framebuffer_step(framebuffer)
display_proxy.add_particle_shading_program(solver.particle_x,
                                           particle_normalized_attr,
                                           colormap_tex,
                                           solver.particle_radius, solver)
display_proxy.add_pile_shading_program(pile)
display_proxy.add_show_framebuffer_shader(framebuffer)
display_proxy.resize(800, 600)

timestamp_str, timestamp_hash = get_timestamp_and_hash()
frame_directory = f'{args.output_dir}/rltruth-{timestamp_hash}-{timestamp_str}'
Path(frame_directory).mkdir(parents=True, exist_ok=True)
# =========== save config
np.save(f'{frame_directory}/seed.npy', args.seed)
np.save(f'{frame_directory}/num_buoys.npy', args.num_buoys)
np.save(f'{frame_directory}/fluid_mass.npy',
        unit.to_real_mass(num_particles * particle_mass))
np.save(f'{frame_directory}/density0_real.npy', density0_real)
np.save(f'{frame_directory}/agitator_option.npy', agitator_option)
np.save(f'{frame_directory}/kinematic_viscosity_real.npy',
        kinematic_viscosity_real)
real_sample_x = dp.create_coated_like(sampling.sample_x)
real_sample_x.set_from(sampling.sample_x)
real_sample_x.scale(unit.to_real_length(1))
real_sample_x.write_file(f'{frame_directory}/sample-x.alu',
                         sampling.num_samples)
dp.remove(real_sample_x)
# =========== end of save config


class OrnsteinUhlenbeckProcess:

    def __init__(self, dim, mu=0.0, sigma=0.2, theta=0.15):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dim = dim
        self.x_prev = np.zeros(dim)

    def __call__(self, dt):
        x = self.x_prev + self.theta * (
            self.mu - self.x_prev
        ) * dt + self.sigma * np.sqrt(dt) * np.random.normal(size=self.dim)
        self.x_prev = x
        return x


linear_acc_rp = OrnsteinUhlenbeckProcess(3,
                                         sigma=np.array([8, 2, 8]),
                                         theta=0.35)
angular_acc_rp = OrnsteinUhlenbeckProcess(3, sigma=10 * np.pi)

truth_real_freq = 100.0
truth_real_interval = 1.0 / truth_real_freq
next_truth_frame_id = 0

visual_real_freq = 30.0
visual_real_interval = 1.0 / visual_real_freq
next_visual_frame_id = 0
visual_x_scaled = dp.create_coated_like(solver.particle_x)
visual_v2_scaled = dp.create_coated_like(solver.particle_cfl_v2)

target_t = unit.from_real_time(10.0)
last_tranquillized = 0.0
rest_state_achieved = False

next_force_time = 0.0
remaining_force_time = 0.0

v_rms = 0
while not rest_state_achieved or solver.t < target_t:
    dp.map_graphical_pointers()
    for frame_interstep in range(10):
        if rest_state_achieved:
            if solver.t >= unit.from_real_time(
                    next_truth_frame_id * truth_real_interval):
                sampling.prepare_neighbor_and_boundary(runner, solver)
                sample_v = sampling.sample_velocity(runner, solver)
                sample_v.scale(unit.to_real_velocity(1))
                sample_v.write_file(
                    f'{frame_directory}/v-{next_truth_frame_id}.alu',
                    sampling.num_samples)
                sample_density = sampling.sample_density(runner)
                sample_density.scale(unit.to_real_density(1))
                sample_density.write_file(
                    f'{frame_directory}/density-{next_truth_frame_id}.alu',
                    sampling.num_samples)
                pile.write_file(
                    f'{frame_directory}/{next_truth_frame_id}.pile',
                    unit.to_real_length(1), unit.to_real_velocity(1),
                    unit.to_real_angular_velocity(1))
                next_truth_frame_id += 1
            if args.write_visual and solver.t >= unit.from_real_time(
                    next_visual_frame_id * visual_real_interval):
                #framebuffer.write(f"{frame_directory}
                pixels = np.array(framebuffer.get(), dtype=np.byte)
                pil_image = Image.fromarray(
                    pixels.reshape(framebuffer.height, framebuffer.width, 3),
                    "RGB")
                pil_image.save(
                    f"{frame_directory}/visual-{next_visual_frame_id}.png")
                visual_x_scaled.set_from(solver.particle_x)
                visual_x_scaled.scale(unit.to_real_length(1))
                visual_x_scaled.write_file(
                    f'{frame_directory}/visual-x-{next_visual_frame_id}.alu',
                    solver.num_particles)
                visual_x_scaled.write_file(
                    f'{frame_directory}/visual-pellets-{next_visual_frame_id}.alu',
                    pile.num_pellets, solver.max_num_particles)
                visual_v2_scaled.set_from(solver.particle_cfl_v2)
                visual_v2_scaled.scale(unit.to_real_velocity_mse(1))
                visual_v2_scaled.write_file(
                    f'{frame_directory}/visual-v2-{next_visual_frame_id}.alu',
                    solver.num_particles)
                pile.write_file(
                    f'{frame_directory}/visual-{next_visual_frame_id}.pile',
                    unit.to_real_length(1), unit.to_real_velocity(1),
                    unit.to_real_angular_velocity(1))
                next_visual_frame_id += 1
            ###### move object #########
            agitator_v_al = pile.v[agitator_id]
            agitator_omega_al = pile.omega[agitator_id]
            agitator_v = np.array(
                [agitator_v_al.x, agitator_v_al.y, agitator_v_al.z],
                dp.default_dtype)
            agitator_omega = np.array([
                agitator_omega_al.x, agitator_omega_al.y, agitator_omega_al.z
            ], dp.default_dtype)

            agitator_x = pile.x[agitator_id]
            agitator_angular_acc = unit.from_real_angular_acceleration(
                angular_acc_rp(solver.dt))
            agitator_a = unit.from_real_acceleration(linear_acc_rp(solver.dt))
            # print('start force', agitator_a, agitator_angular_acc)

            agitator_v += agitator_a * solver.dt
            agitator_omega += agitator_angular_acc * solver.dt

            pile.v[agitator_id] = dp.f3(agitator_v[0], agitator_v[1],
                                        agitator_v[2])
            pile.omega[agitator_id] = dp.f3(agitator_omega[0],
                                            agitator_omega[1],
                                            agitator_omega[2])
        else:  # if not rest_state_achieved
            v_rms = np.sqrt(
                runner.sum(solver.particle_cfl_v2, solver.num_particles) /
                solver.num_particles)
            if unit.to_real_time(solver.t - last_tranquillized) > 0.45:
                solver.particle_v.set_zero()
                solver.reset_solving_var()
                last_tranquillized = solver.t
            elif unit.to_real_time(solver.t - last_tranquillized
                                   ) > 0.4 and unit.to_real_velocity(
                                       v_rms) < 0.030:
                print("rest state achieved at", unit.to_real_time(solver.t))
                solver.t = 0
                rest_state_achieved = True
        solver.step()
    print(
        f"t = {unit.to_real_time(solver.t) } dt = {unit.to_real_time(solver.dt)} cfl = {solver.utilized_cfl} vrms={unit.to_real_velocity(v_rms)} max_v={unit.to_real_velocity(np.sqrt(solver.max_v2))} num solves = {solver.num_density_solve}"
    )
    solver.normalize(solver.particle_v, particle_normalized_attr, 0,
                     unit.from_real_velocity(0.01))
    dp.unmap_graphical_pointers()
    display_proxy.draw()

dp.remove(particle_normalized_attr)
dp.remove(visual_x_scaled)
dp.remove(visual_v2_scaled)
del solver
del pile
del runner
del dp

if args.render:
    subprocess.Popen([
        "blender",
        "-b",
        "/home/kennychufk/workspace/blenderWs/alluvion-film/liquid-glyph-cycle-new.blend",
        "--python",
        "/home/kennychufk/workspace/blenderWs/alluvion-film/glyph-vis.py",
        "--",
        "-s",
        "0",
        "-e",
        "300",
        "-d",
        f"/home/kennychufk/workspace/pythonWs/test-run-al-outside/{frame_directory}",
        "--output-prefix",
        "truth-agitator",
        "--render-liquid",
        "1",
        "--render-glyph",
        "0",
        "--use-cylinder-buoy",
        "0",
        "--render-buoy",
        "1",
        "--render-buoy-label",
        "0",
        "--render-agitator",
        "1",
        "--render-beads",
        "0",
    ],
                     env=os.environ.copy()).wait()
