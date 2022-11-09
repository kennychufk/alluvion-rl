import argparse
import numpy as np

import alluvol

from util import read_alu

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)
args = parser.parse_args()

num_beads = len(read_alu(f'{args.input}/x-0.alu'))
print('num_beads', num_beads)

raster_radius = (2**-8) * 0.36
voxel_size = raster_radius / np.sqrt(3.0)

recon_raster_radius = (2**-6) * 0.36
recon_voxel_size = recon_raster_radius / np.sqrt(3.0)

for i in range(1000):
    ls = alluvol.create_liquid_level_set(read_alu(f'{args.input}/x-{i}.alu'),
                                         raster_radius, voxel_size)
    ls.write(f'{args.input}/x-{i}.vdb')

    ls_resampled = ls.resample(recon_voxel_size)
    ls_resampled.write(f'{args.input}/x-{i}-resampled6.vdb')
