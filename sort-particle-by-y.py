import alluvion as al
import numpy as np

dp = al.Depot(np.float32)
particle_x_filename = 'initial_bead_x-2to-8.alu'
particle_v_filename = 'initial_bead_v-2to-8.alu'
num_particles = dp.get_alu_info(particle_x_filename)[0][0]

particle_x = dp.create_coated((num_particles), 3)
particle_x.read_file(particle_x_filename)
particle_x_np = particle_x.get()

particle_v = dp.create_coated((num_particles), 3)
particle_v.read_file(particle_v_filename)
particle_v_np = particle_v.get()

sorted_indices = np.argsort(particle_x_np[:, 1])
sorted_x_np = particle_x_np[sorted_indices]
sorted_v_np = particle_v_np[sorted_indices]
print(sorted_x_np)

particle_x.set(sorted_x_np)
particle_x.write_file('sorted-bead-x-2to-8.alu', num_particles)
particle_v.set(sorted_v_np)
particle_v.write_file('sorted-bead-v-2to-8.alu', num_particles)
