from util import read_alu

import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str)
parser.add_argument('output', type=str)

args = parser.parse_args()

particle_x_np = read_alu(args.input)
num_particles = len(particle_x_np)

with open(args.output, 'w') as f:
    for i in range(num_particles):
        x = particle_x_np[i]
        f.write(f'v {x[0]} {x[1]} {x[2]}\n')
