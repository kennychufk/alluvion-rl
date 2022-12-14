import alluvion as al
import numpy as np
import argparse

import alluvol

parser = argparse.ArgumentParser(description='Silhouette cutter')
parser.add_argument('--shape-dir', type=str, required=True)
args = parser.parse_args()

raster_radius = (2**-6) * 0.36
voxel_size = raster_radius / np.sqrt(3.0)
agitator_exclusion_dist = 0.06

agitator_names = [
    "02942699/9db4b2c19c858a36eb34db531a289b8e",
    "03046257/757fd88d3ddca2403406473757712946",
    "03261776/1d4f9c324d6388a9b904f4192b538029",
    "03325088/daa26435e1603392b7a867e9b35a1295",
    "03513137/91c0193d38f0c5338c9affdacaf55648",
    "03636649/83353863ea1349682ebeb1e6a8111f53",
    "03759954/35f36e337df50fcb92b3c4741299c1af",
    "03797390/ec846432f3ebedf0a6f32a8797e3b9e9",
    "04401088/e5bb37415abcf3b7e1c1039f91f43fda",
    "04530566/66a90b7b92ff2549f2635cfccf45023", "bunny/bunny",
    "stirrer/stirrer"
]

for agitator_name in agitator_names:
    ls = alluvol.create_mesh_level_set(
        f"{args.shape_dir}/{agitator_name}/models/manifold2-decimate-2to-8.obj",
        voxel_size,
        scale=0.00390625,
        ex_band=int(agitator_exclusion_dist / voxel_size) + 3)
    ls.write_obj(
        f"{args.shape_dir}/{agitator_name}/models/manifold2-decimate-pa-dilate.obj",
        agitator_exclusion_dist)
