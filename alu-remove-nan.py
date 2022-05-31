from util import read_alu
import alluvion as al

import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str)
parser.add_argument('output', type=str)

args = parser.parse_args()

particle_x_np = read_alu(args.input)
num_particles = len(particle_x_np)

particle_x_filtered = np.zeros_like(particle_x_np, dtype=particle_x_np.dtype)

num_valid = 0
for i in range(num_particles):
    x = particle_x_np[i]
    if (np.sum(np.isnan(x))== 0):
        particle_x_filtered[num_valid] = x
        num_valid += 1


dp = al.Depot(np.float32)
print('num_valid', num_valid)
x_filtered_al = dp.create_coated((num_valid), 3, particle_x_np.dtype)
x_filtered_al.set(particle_x_filtered[:num_valid])
x_filtered_al.write_file(args.output, num_valid)
