import numpy as np


class FluidSample:
    def __init__(self, dp, x_src):
        set_with_file = isinstance(x_src, str)
        self.num_samples = dp.get_alu_info(
            x_src)[0][0] if set_with_file else x_src.shape[0]

        self.sample_x = dp.create_coated((self.num_samples), 3)
        self.sample_data3 = dp.create_coated((self.num_samples), 3)
        self.sample_data1 = dp.create_coated((self.num_samples), 1)
        self.sample_boundary = dp.create_coated(
            (dp.cni.num_boundaries, self.num_samples), 4)
        self.sample_boundary_kernel = dp.create_coated(
            (dp.cni.num_boundaries, self.num_samples), 4)
        self.sample_neighbors = dp.create_coated(
            (self.num_samples, dp.cni.max_num_neighbors_per_particle), 4)
        self.sample_num_neighbors = dp.create_coated((self.num_samples), 1,
                                                     np.uint32)
        if set_with_file:
            self.sample_x.read_file(x_src)
        else:
            self.sample_x.set(x_src)
        self.dp = dp

    def destroy_variables(self):
        self.dp.remove(self.sample_x)
        self.dp.remove(self.sample_data3)
        self.dp.remove(self.sample_data1)
        self.dp.remove(self.sample_boundary)
        self.dp.remove(self.sample_boundary_kernel)
        self.dp.remove(self.sample_neighbors)
        self.dp.remove(self.sample_num_neighbors)

    def prepare_neighbor_and_boundary(self, runner, solver):
        solver.update_particle_neighbors()
        runner.launch_make_neighbor_list(self.sample_x, solver.pid,
                                         solver.pid_length,
                                         self.sample_neighbors,
                                         self.sample_num_neighbors,
                                         self.num_samples)
        solver.sample_all_boundaries(self.sample_x, self.sample_boundary,
                                     self.sample_boundary_kernel,
                                     self.num_samples)

    def prepare_neighbor_and_boundary_wrap1(self, runner, solver):
        solver.update_particle_neighbors_wrap1()
        runner.launch_make_neighbor_list_wrap1(self.sample_x, solver.pid,
                                               solver.pid_length,
                                               self.sample_neighbors,
                                               self.sample_num_neighbors,
                                               self.num_samples)
        solver.sample_all_boundaries(self.sample_x, self.sample_boundary,
                                     self.sample_boundary_kernel,
                                     self.num_samples)

    def sample_vector3(self, runner, solver, fluid_var):
        runner.launch_sample_fluid(self.sample_x, solver.particle_x,
                                   solver.particle_density, fluid_var,
                                   self.sample_neighbors,
                                   self.sample_num_neighbors,
                                   self.sample_data3, self.num_samples)
        return self.sample_data3

    def sample_density(self, runner):
        runner.launch_sample_density(self.sample_x, self.sample_neighbors,
                                     self.sample_num_neighbors,
                                     self.sample_data1,
                                     self.sample_boundary_kernel,
                                     self.num_samples)
        return self.sample_data1

    def sample_velocity(self, runner, solver):
        runner.launch_sample_velocity(
            self.sample_x, solver.particle_x, solver.particle_density,
            solver.particle_v, self.sample_neighbors,
            self.sample_num_neighbors, self.sample_data3, self.sample_boundary,
            self.sample_boundary_kernel, solver.pile.x_device,
            solver.pile.v_device, solver.pile.omega_device, self.num_samples)
        return self.sample_data3
