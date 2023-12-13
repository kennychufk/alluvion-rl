import os
import subprocess
from pathlib import Path
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s",
                    "--start",
                    type=int,
                    default=0,
                    help="Start frame id (30Hz)")
parser.add_argument("-e", "--end", type=int, default=-1, help="End frame id")
parser.add_argument("-d", "--directory", required=True, help="scene directory")
parser.add_argument("-r",
                    "--recon-directory",
                    type=str,
                    default="",
                    help="reconstruction directory")
parser.add_argument("--render-agitator", type=int, default=0)
parser.add_argument("--empirical-scene", type=int, default=0)
args = parser.parse_args()


def create_fluid_mesh(scene_dir, particle_radius, frame_id, container_filename,
                      buoy_filename, agitator_filename, agitator_scale):
    frame_id_100 = int(frame_id / 30 * 100)
    output_filename = f"{scene_dir}/{frame_id_100}.obj"
    exec_tokens = [
        'allumesh', f"{particle_radius:.18g}",
        f"{scene_dir}/x-{frame_id_100}.alu",
        f"{scene_dir}/{frame_id_100}.pile", container_filename, buoy_filename,
        agitator_filename, f"{agitator_scale:.18g}", output_filename
    ]
    print(" ".join(exec_tokens))
    if not Path(output_filename).is_file():
        subprocess.Popen(exec_tokens, env=os.environ.copy()).wait()
    return output_filename


frame_start = args.start
frame_end = args.end if args.end >= 0 else 300

render_reconstruction_result = (len(args.recon_directory) > 0)
render_agitator = (args.render_agitator == 1)
empirical_scene = args.empirical_scene  # 0: not empirical, 1: on IM, 2: on tray
scene_dir = args.directory
target_dir = args.recon_directory if render_reconstruction_result else scene_dir

real_kernel_radius_truth = 2**-8
real_kernel_radius_reconstruction = 0.015
real_kernel_radius = real_kernel_radius_reconstruction if render_reconstruction_result else real_kernel_radius_truth
particle_radius = real_kernel_radius * 0.36
target_dir = args.recon_directory if render_reconstruction_result else scene_dir

shape_dir = str(Path.home().joinpath('workspace/pythonWs/shape-al'))
container_filename = f"{shape_dir}/cube24/cube24/models/manifold2-decimate-pa-elevated.obj" if empirical_scene else f"{shape_dir}/cube24/cube24/models/manifold2-decimate-pa.obj"

agitator_option_filename = f"{scene_dir}/agitator_option.npy"
if Path(agitator_option_filename).exists():
    agitator_option = np.load(agitator_option_filename).item()
    agitator_scale_real = real_kernel_radius_truth
    agitator_filename = str(
        Path(
            f"{shape_dir}/{agitator_option}/models/manifold2-decimate-2to-8.obj"
        ).absolute())
buoy_cylinder_filename = str(
    Path(f"{shape_dir}/buoy/buoy/models/manifold2-decimate-pa.obj").absolute())

for frame_id in range(frame_start, frame_end):
    fluid_mesh_filename = create_fluid_mesh(
        target_dir, particle_radius, frame_id, container_filename,
        "0" if render_reconstruction_result else buoy_cylinder_filename,
        "0" if render_reconstruction_result or not render_agitator else
        agitator_filename, 0 if render_reconstruction_result
        or not render_agitator else agitator_scale_real)
